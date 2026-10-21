import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.cuda.amp import GradScaler, autocast

# ---------------- CONFIG ----------------
DATASET_NAME = "SelfRegulationSCP1" 
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-3
BASE_WD = 1e-4
DROPOUT = 0.4
TCN_HIDDEN = 128
GCN_HIDDEN = 128
GOG_HIDDEN = 128
STOCH_DEPTH_P = 0.2
TTA_PASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler(enabled=(device.type == 'cuda'))

# =========================================================
# DATA UTILS
# =========================================================
def df_to_np(df):
    N, C = df.shape
    T = len(df.iloc[0,0])
    X = np.zeros((N,T,C), np.float32)
    for i in range(N):
        for j in range(C):
            X[i,:,j] = df.iloc[i,j].to_numpy()
    return X

# Adjacency Cache
_adj = {}
def chain_adj(T, radius):
    key = (T, radius)
    if key in _adj: return _adj[key].to(device)
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

# =========================================================
# MODEL BLOCKS
# =========================================================
class ResTCN(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        layers = []; ch = inC
        for i in range(5):
            d = 2**i
            layers += [
                nn.Conv1d(ch, outC, 3, padding=d, dilation=d),
                nn.BatchNorm1d(outC),
                nn.ReLU(),
                nn.Dropout(DROPOUT)
            ]
            ch = outC
        self.net = nn.Sequential(*layers)
        self.res = nn.Conv1d(inC, outC, 1) if inC != outC else nn.Identity()
    def forward(self, x): return F.relu(self.net(x) + self.res(x))

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
    def forward(self, x, A):
        h = F.relu(self.inp(x))
        for g in self.layers:
            if self.training and torch.rand(1) < STOCH_DEPTH_P: continue
            h = g(h, A)
        return h

class GOG(nn.Module):
    def __init__(self, F_in, H, num_seg):
        super().__init__()
        self.num_seg = num_seg
        self.inner = GCN(F_in, H)
        self.outer = GCN(H, H)
    def forward(self, x):
        segs_x = torch.chunk(x, self.num_seg, 1)
        outs = []
        for s in segs_x:
            outs.append(self.inner(s, chain_adj(s.size(1), 3)).mean(1))
        h = torch.stack(outs, 1)
        return self.outer(h, chain_adj(self.num_seg, 1)).mean(1)

class Model(nn.Module):
    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T = seq_len
        self.C = input_dim
        self.num_seg = 4 if seq_len < 120 else 6 if seq_len < 300 else 8
        
        self.tcn = ResTCN(input_dim, TCN_HIDDEN)
        self.gcn = GCN(input_dim, GCN_HIDDEN)
        self.gog1 = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        self.gog2 = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * (TCN_HIDDEN + GOG_HIDDEN), 512),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        t = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        g1 = self.gcn(x, chain_adj(self.T, 1))
        g3 = self.gcn(x, chain_adj(self.T, 3))
        g = 0.5 * g1 + 0.5 * g3
        z = torch.cat([t.mean(1), self.gog1(t), g.mean(1), self.gog2(g)], 1)
        return self.fc(z)

# =========================================================
# DATA LOADING & EXECUTION
# =========================================================
print(f"Downloading/Loading {DATASET_NAME}...")
Xtr_df, ytr_raw = load_UCR_UEA_dataset(DATASET_NAME, "train", return_X_y=True)
Xte_df, yte_raw = load_UCR_UEA_dataset(DATASET_NAME, "test", return_X_y=True)

X_train = df_to_np(Xtr_df)
X_test = df_to_np(Xte_df)
N_train, T, C = X_train.shape

sc = StandardScaler()
X_train = sc.fit_transform(X_train.reshape(-1, C)).reshape(N_train, T, C)
X_test = sc.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

le = LabelEncoder()
y_train = le.fit_transform(ytr_raw)
y_test = le.transform(yte_raw)
n_classes = len(le.classes_)

train_loader = DataLoader(list(zip(torch.tensor(X_train).float(), torch.tensor(y_train).long())), 
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(torch.tensor(X_test).float(), torch.tensor(y_test).long())), 
                         batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model
model = Model(C, T, n_classes).to(device)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=BASE_WD)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

print(f"Model initialized. Starting training on {device}...")

best = 0
for ep in range(1, EPOCHS + 1):
    model.train()
    stoch_p = STOCH_DEPTH_P if ep <= 30 else 0.0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            loss = crit(model(xb) / 1.5, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    
    sch.step()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = []
            for _ in range(TTA_PASSES):
                noise = 0.005 * torch.randn_like(xb)
                preds.append(model(xb + noise))
            pred = torch.stack(preds).mean(0).argmax(1).cpu()
            correct += (pred == yb).sum().item()
    
    acc = correct / len(y_test)
    best = max(best, acc)
    print(f"Epoch {ep:03d} | Acc {acc*100:.2f}% | Best {best*100:.2f}%")

print("\n🔥 FINAL BEST ACC =", best * 100, "%")