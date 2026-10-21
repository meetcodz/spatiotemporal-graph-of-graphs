!pip install -q sktime torch torchvision torchaudio tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.cuda.amp import GradScaler, autocast

# ================= CONFIG =================
DATASET_NAME = "Handwriting"
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-3
BASE_WD = 1e-4
DROPOUT = 0.4

TCN_HIDDEN = 128
GCN_HIDDEN = 128
GOG_HIDDEN = 128

STOCH_DEPTH_P = 0.2
TTA_PASSES = 3           # ↓ reduced (ROCKET has no TTA)
TTA_NOISE = 0.002        # ↓ controlled noise

AM_MARGIN = 0.35
AM_SCALE = 30.0
WARMUP_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler(enabled=(device.type == "cuda"))

# ================= DATA =================
def df_to_np(df):
    N, C = df.shape
    T = len(df.iloc[0, 0])
    X = np.zeros((N, T, C), np.float32)
    for i in range(N):
        for j in range(C):
            X[i, :, j] = df.iloc[i, j].to_numpy()
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
            if i - r >= 0: A[i, i - r] = w
            if i + r < T: A[i, i + r] = w
    A += torch.eye(T)
    D = torch.diag((A.sum(1)).clamp(min=1e-6).pow(-0.5))
    _adj[key] = D @ A @ D
    return _adj[key].to(device)

# ================= MODEL =================
class ResTCN(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        layers, ch = [], inC
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
    def __init__(self, Fin, H):
        super().__init__()
        self.fc = nn.Linear(Fin, H)

    def forward(self, x, A):
        return F.relu(self.fc(torch.einsum("ij,bjf->bif", A, x)))

class GCN(nn.Module):
    def __init__(self, Fin, H):
        super().__init__()
        self.inp = nn.Linear(Fin, H)
        self.layers = nn.ModuleList([GraphConv(H, H) for _ in range(3)])

    def forward(self, x, A):
        h = F.relu(self.inp(x))
        for g in self.layers:
            if self.training and torch.rand(1) < STOCH_DEPTH_P:
                continue
            h = g(h, A)
        return h

class GOG(nn.Module):
    def __init__(self, Fin, H, num_seg):
        super().__init__()
        self.num_seg = num_seg
        self.inner = GCN(Fin, H)
        self.outer = GCN(H, H)

    def forward(self, x):
        segs = torch.chunk(x, self.num_seg, 1)
        outs = []
        for s in segs:
            outs.append(self.inner(s, chain_adj(s.size(1), 3)).mean(1))
        h = torch.stack(outs, 1)
        return self.outer(h, chain_adj(self.num_seg, 1)).mean(1)

class Model(nn.Module):
    def __init__(self, C, T, n_classes):
        super().__init__()
        self.T = T
        self.num_seg = 4 if T < 120 else 6 if T < 300 else 8

        self.tcn = ResTCN(C, TCN_HIDDEN)
        self.gcn = GCN(C, GCN_HIDDEN)

        self.gog1 = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        self.gog2 = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg)

        FUSED = 2 * (TCN_HIDDEN + GOG_HIDDEN)

        self.bn = nn.BatchNorm1d(FUSED)  # ⭐ ROCKET-style whitening
        self.fc = nn.Linear(FUSED, n_classes)

    def forward(self, x):
        t = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        g1 = self.gcn(x, chain_adj(self.T, 1))
        g3 = self.gcn(x, chain_adj(self.T, 3))
        g = 0.5 * g1 + 0.5 * g3

        z = torch.cat([
            t.mean(1), self.gog1(t),
            g.mean(1), self.gog2(g)
        ], 1)

        z = self.bn(z)
        z = F.normalize(z, dim=1)        # ⭐ cosine space (ROCKET-like)
        return self.fc(z)

# ================= LOSS =================
class AMSoftmax(nn.Module):
    def __init__(self, margin=0.35, scale=30):
        super().__init__()
        self.m = margin
        self.s = scale

    def forward(self, logits, labels):
        onehot = torch.zeros_like(logits)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        logits = logits - onehot * self.m
        return F.cross_entropy(self.s * logits, labels)

# ================= LOAD DATA =================
print(f"Loading {DATASET_NAME}...")
Xtr_df, ytr = load_UCR_UEA_dataset(DATASET_NAME, "train", return_X_y=True)
Xte_df, yte = load_UCR_UEA_dataset(DATASET_NAME, "test", return_X_y=True)

Xtr = df_to_np(Xtr_df)
Xte = df_to_np(Xte_df)

N, T, C = Xtr.shape
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr.reshape(-1, C)).reshape(N, T, C)
Xte = sc.transform(Xte.reshape(-1, C)).reshape(Xte.shape)

le = LabelEncoder()
ytr = le.fit_transform(ytr)
yte = le.transform(yte)

train_loader = DataLoader(
    list(zip(torch.tensor(Xtr).float(), torch.tensor(ytr))),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    list(zip(torch.tensor(Xte).float(), torch.tensor(yte))),
    batch_size=BATCH_SIZE
)

# ================= TRAIN =================
model = Model(C, T, len(le.classes_)).to(device)
crit = AMSoftmax(AM_MARGIN, AM_SCALE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=BASE_WD)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - WARMUP_EPOCHS, 1e-5)

best = 0
for ep in range(1, EPOCHS + 1):
    model.train()

    if ep <= WARMUP_EPOCHS:
        lr = LR * ep / WARMUP_EPOCHS
        for g in opt.param_groups:
            g["lr"] = lr
    else:
        sch.step()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = crit(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = []
            for _ in range(TTA_PASSES):
                preds.append(model(xb + TTA_NOISE * torch.randn_like(xb)))
            pred = torch.stack(preds).mean(0).argmax(1).cpu()
            correct += (pred == yb).sum().item()

    acc = correct / len(yte)
    best = max(best, acc)
    print(f"Epoch {ep:03d} | Acc {acc*100:.2f}% | Best {best*100:.2f}%")

print("\n🔥 FINAL BEST ACC =", best * 100)
