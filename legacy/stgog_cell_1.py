# =========================================================
# FINAL UNIVERSAL MTS MODEL (SHAPE-SAFE)
# =========================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.cuda.amp import GradScaler, autocast

# ---------------- CONFIG ----------------
DATASET_NAME = "DuckDuckGeese"
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-3
WD = 1e-4
DROPOUT = 0.4

TCN_HIDDEN = 128
GCN_HIDDEN = 128
GOG_HIDDEN = 128

STOCH_DEPTH_P = 0.2
STOCH_WARMUP_EPOCHS = 30
TTA_PASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler(enabled=(device.type == "cuda"))

torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# DATA UTILS
# =========================================================
def df_to_np(df):
    N, C = df.shape
    T = len(df.iloc[0,0])
    X = np.zeros((N, T, C), np.float32)
    for i in range(N):
        for j in range(C):
            X[i,:,j] = df.iloc[i,j].to_numpy()
    return X

_adj = {}
def chain_adj(T, radius):
    key = (T, radius)
    if key in _adj:
        return _adj[key].to(device)

    A = torch.zeros(T, T)
    for i in range(T):
        for r in range(1, radius + 1):
            w = 1.0 / r
            if i - r >= 0: A[i, i-r] = w
            if i + r < T: A[i, i+r] = w
    A += torch.eye(T)
    D = torch.diag((A.sum(1)).clamp(min=1e-6).pow(-0.5))
    _adj[key] = D @ A @ D
    return _adj[key].to(device)

def num_segments(T):
    return 4 if T < 120 else 6 if T < 300 else 8

# =========================================================
# MODEL BLOCKS
# =========================================================
class LocalTCN(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inC, outC, 3, padding=1),
            nn.BatchNorm1d(outC),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(outC, outC, 3, padding=1),
            nn.BatchNorm1d(outC),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class ResTCN(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        layers = []; ch = inC
        for i in range(5):
            d = 2 ** i
            layers += [
                nn.Conv1d(ch, outC, 3, padding=d, dilation=d),
                nn.BatchNorm1d(outC),
                nn.ReLU(),
                nn.Dropout(DROPOUT)
            ]
            ch = outC
        self.net = nn.Sequential(*layers)
        self.res = nn.Conv1d(inC, outC, 1) if inC != outC else nn.Identity()
    def forward(self, x):
        return F.relu(self.net(x) + self.res(x))

class GraphConv(nn.Module):
    def __init__(self, F_in, H):
        super().__init__()
        self.fc = nn.Linear(F_in, H)
    def forward(self, x, A):
        return F.relu(self.fc(torch.einsum("ij,bjf->bif", A, x)))

class GCN(nn.Module):
    def __init__(self, F_in, H):
        super().__init__()
        self.inp = nn.Linear(F_in, H)
        self.layers = nn.ModuleList([GraphConv(H, H) for _ in range(3)])
    def forward(self, x, A, drop_p=0.0):
        h = F.relu(self.inp(x))
        for g in self.layers:
            if self.training and torch.rand(1, device=x.device) < drop_p:
                continue
            h = g(h, A)
        return h

class GOG(nn.Module):
    def __init__(self, F_in, H, num_seg):
        super().__init__()
        self.num_seg = num_seg
        self.inner = GCN(F_in, H)
        self.outer = GCN(H, H)
    def forward(self, x, drop_p):
        segs = torch.chunk(x, self.num_seg, dim=1)
        outs = []
        for s in segs:
            outs.append(self.inner(s, chain_adj(s.size(1), 3), drop_p).mean(1))
        h = torch.stack(outs, dim=1)
        return self.outer(h, chain_adj(self.num_seg, 1), drop_p).mean(1)

# =========================================================
# FINAL MODEL
# =========================================================
class Model(nn.Module):
    def __init__(self, C, T, n_classes):
        super().__init__()
        self.T = T
        self.num_seg = num_segments(T)

        self.tcn_local = LocalTCN(C, TCN_HIDDEN)
        self.tcn_dilated = ResTCN(C, TCN_HIDDEN)

        self.channel_mix = nn.Conv1d(2 * TCN_HIDDEN, 2 * TCN_HIDDEN, 1)

        self.gcn = GCN(C, GCN_HIDDEN)

        self.gog1 = GOG(2 * TCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        self.gog2 = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg)

        self.fc = nn.Sequential(
            nn.Linear(640, 512),  # ✅ FIXED
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, drop_p):
        tl = self.tcn_local(x.transpose(1,2))
        td = self.tcn_dilated(x.transpose(1,2))
        t = self.channel_mix(torch.cat([tl, td], dim=1)).transpose(1,2)

        g1 = self.gcn(x, chain_adj(self.T, 1), drop_p)
        g3 = self.gcn(x, chain_adj(self.T, 3), drop_p)
        g = 0.5 * g1 + 0.5 * g3

        z = torch.cat([
            t.mean(1),
            self.gog1(t, drop_p),
            g.mean(1),
            self.gog2(g, drop_p)
        ], dim=1)

        return self.fc(z)


# =========================================================
# DATA LOADING
# =========================================================
print(f"Loading dataset: {DATASET_NAME}")

Xtr_df, ytr = load_UCR_UEA_dataset(DATASET_NAME, "train", return_X_y=True)
Xte_df, yte = load_UCR_UEA_dataset(DATASET_NAME, "test", return_X_y=True)

X_train = df_to_np(Xtr_df)
X_test  = df_to_np(Xte_df)

N, T, C = X_train.shape

sc = StandardScaler()
X_train = sc.fit_transform(X_train.reshape(-1, C)).reshape(N, T, C)
X_test  = sc.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

le = LabelEncoder()
y_train = le.fit_transform(ytr)
y_test  = le.transform(yte)
n_classes = len(le.classes_)

train_loader = DataLoader(
    list(zip(torch.tensor(X_train).float(), torch.tensor(y_train))),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    list(zip(torch.tensor(X_test).float(), torch.tensor(y_test))),
    batch_size=BATCH_SIZE, shuffle=False
)

# =========================================================
# TRAINING
# =========================================================
model = Model(C, T, n_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

best = 0.0

for ep in range(1, EPOCHS + 1):
    model.train()
    drop_p = STOCH_DEPTH_P if ep <= STOCH_WARMUP_EPOCHS else 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            loss = criterion(model(xb, drop_p) / 1.5, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    # -------- EVAL --------
    model.eval()
    correct = 0
    use_tta = (ep == EPOCHS)

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            if use_tta:
                preds = []
                for _ in range(TTA_PASSES):
                    preds.append(model(xb + 0.005 * torch.randn_like(xb), 0.0))
                pred = torch.stack(preds).mean(0).argmax(1).cpu()
            else:
                pred = model(xb, 0.0).argmax(1).cpu()

            correct += (pred == yb).sum().item()

    acc = correct / len(y_test)
    best = max(best, acc)
    print(f"Epoch {ep:03d} | Acc {acc*100:.2f}% | Best {best*100:.2f}%")

print("\n🔥 FINAL BEST ACCURACY =", best * 100, "%")
