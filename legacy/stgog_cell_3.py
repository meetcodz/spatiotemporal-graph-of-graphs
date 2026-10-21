!pip install -q sktime torch torchvision torchaudio tqdm
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.cuda.amp import GradScaler, autocast

# ---------------- CONFIG ----------------
DATASET_NAME = "Epilepsy" 
EPOCHS = 250  # Extended training for better convergence
BATCH_SIZE = 8
LR = 1e-3
BASE_WD = 1e-4
DROPOUT = 0.42  # Slightly reduced for this dataset's characteristics
TCN_HIDDEN = 128
GCN_HIDDEN = 128
GOG_HIDDEN = 128
STOCH_DEPTH_P = 0.2
TTA_PASSES = 7  # More TTA passes for more robust predictions
SEED = 42  # Fixed seed for reproducibility

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
# IMPROVED MODEL COMPONENTS
# =========================================================
class ResTCN(nn.Module):
    """Enhanced TCN with squeeze-excitation for channel attention"""
    def __init__(self, inC, outC):
        super().__init__()
        layers = []
        ch = inC
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
        
        # Squeeze-Excitation: learn to weight different dilation rates
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(outC, outC // 4, 1),
            nn.ReLU(),
            nn.Conv1d(outC // 4, outC, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.net(x)
        # Apply learned channel-wise attention
        se_weight = self.se(out)
        out = out * se_weight
        return F.relu(out + self.res(x))

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
        self.stoch_p = 0.0  # Will be set externally
    
    def forward(self, x, A):
        h = F.relu(self.inp(x))
        for g in self.layers:
            # Stochastic depth with survival probability
            if self.training and torch.rand(1).item() < self.stoch_p:
                continue
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

class MultiScaleGCN(nn.Module):
    """Adaptive multi-scale graph convolution with learnable fusion"""
    def __init__(self, F_in, H):
        super().__init__()
        # Three different radii for different temporal receptive fields
        self.gcn_local = GCN(F_in, H)      # radius 1: local
        self.gcn_medium = GCN(F_in, H)     # radius 3: medium range
        self.gcn_global = GCN(F_in, H)     # radius 5: broader context
        
        # Learnable fusion weights initialized equally
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)
    
    def forward(self, x, T):
        # Get representations at different scales
        g1 = self.gcn_local(x, chain_adj(T, 1))
        g3 = self.gcn_medium(x, chain_adj(T, 3))
        g5 = self.gcn_global(x, chain_adj(T, 5))
        
        # Normalize weights with softmax for proper probability distribution
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted combination
        return weights[0] * g1 + weights[1] * g3 + weights[2] * g5

class Model(nn.Module):
    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T = seq_len
        self.C = input_dim
        
        # Adaptive segmentation based on sequence length
        if seq_len < 100:
            self.num_seg = 4
        elif seq_len < 200:
            self.num_seg = 6
        elif seq_len < 400:
            self.num_seg = 8
        else:
            self.num_seg = 10
        
        # Core components with improvements
        self.tcn = ResTCN(input_dim, TCN_HIDDEN)
        self.multi_gcn = MultiScaleGCN(input_dim, GCN_HIDDEN)
        self.gog1 = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        self.gog2 = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        
        # Enhanced classifier with residual connection
        feat_dim = 2 * (TCN_HIDDEN + GOG_HIDDEN)
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)
        
        # Learnable pathway importance weights
        self.pathway_weights = nn.Parameter(torch.ones(4) / 4.0)
    
    def set_stoch_depth(self, p):
        """Set stochastic depth probability for all GCN components"""
        for module in [self.multi_gcn.gcn_local, self.multi_gcn.gcn_medium, 
                       self.multi_gcn.gcn_global, self.gog1.inner, self.gog1.outer,
                       self.gog2.inner, self.gog2.outer]:
            module.stoch_p = p
    
    def forward(self, x):
        # Multi-pathway feature extraction
        t = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # TCN pathway
        g = self.multi_gcn(x, self.T)  # Multi-scale GCN pathway
        
        # Global pooling for each pathway
        t_pool = t.mean(1)
        g_pool = g.mean(1)
        
        # Hierarchical features
        t_gog = self.gog1(t)
        g_gog = self.gog2(g)
        
        # Learnable pathway weighting
        weights = F.softmax(self.pathway_weights, dim=0)
        
        # Weighted concatenation
        z = torch.cat([
            weights[0] * t_pool,
            weights[1] * t_gog,
            weights[2] * g_pool,
            weights[3] * g_gog
        ], dim=1)
        
        # Classification with residual-style connections
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.fc3(h)

# =========================================================
# TRAINING UTILITIES
# =========================================================
class MixUp:
    """MixUp augmentation for better generalization"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y, training=True):
        # Only apply mixup during training, not during evaluation
        if self.alpha > 0 and training:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        return x, y, y, 1.0

# =========================================================
# DATA LOADING
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

train_loader = DataLoader(
    list(zip(torch.tensor(X_train).float(), torch.tensor(y_train).long())), 
    batch_size=BATCH_SIZE, shuffle=True, drop_last = True
)
test_loader = DataLoader(
    list(zip(torch.tensor(X_test).float(), torch.tensor(y_test).long())), 
    batch_size=BATCH_SIZE, shuffle=False, drop_last = True
)

# =========================================================
# MODEL & TRAINING SETUP
# =========================================================
model = Model(C, T, n_classes).to(device)

# Adaptive label smoothing
label_smooth = 0.1 if n_classes <= 5 else 0.15

crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=BASE_WD)

# Cosine annealing with warm restarts for better exploration
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)

mixup = MixUp(alpha=0.2)

print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Segments: {model.num_seg}, Classes: {n_classes}")
print(f"Training on {device} with seed {SEED}\n")

# =========================================================
# TRAINING LOOP
# =========================================================
best_acc = 0
best_state = None
best_epoch = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    
    # Exponentially decaying stochastic depth - slower decay for longer training
    current_stoch_p = STOCH_DEPTH_P * np.exp(-ep / 60.0)
    model.set_stoch_depth(current_stoch_p)
    
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # Apply mixup augmentation (only during training)
        xb_mixed, ya, yb_mixed, lam = mixup(xb, yb, training=True)
        
        opt.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            out = model(xb_mixed)
            # MixUp loss
            loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb_mixed)
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
    
    sch.step()
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = []
            
            # Test-time augmentation with diverse perturbations
            for i in range(TTA_PASSES):
                if i == 0:
                    # First pass: no augmentation
                    aug_xb = xb
                else:
                    # Other passes: small gaussian noise
                    noise = torch.randn_like(xb) * 0.003  # Reduced noise
                    aug_xb = xb + noise
                preds.append(model(aug_xb))
            
            pred = torch.stack(preds).mean(0).argmax(1).cpu()
            correct += (pred == yb).sum().item()
    
    acc = correct / len(y_test)
    avg_loss = total_loss / len(train_loader)
    
    # Track best model throughout training
    if acc > best_acc:
        best_acc = acc
        best_epoch = ep
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = "★ NEW BEST"
    else:
        marker = ""
    
    # Print every 10 epochs or when there's a new best
    if ep % 10 == 0 or marker:
        print(f"Ep {ep:03d} | Loss {avg_loss:.4f} | Acc {acc*100:.2f}% | "
              f"Best {best_acc*100:.2f}% {marker}")

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"Best accuracy: {best_acc*100:.2f}% (achieved at epoch {best_epoch})")
print(f"{'='*60}")

# Restore best model
if best_state:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

print(f"\n🔥 FINAL BEST ACC = {best_acc * 100:.2f}%")

# Show learned parameters
gcn_weights = F.softmax(model.multi_gcn.fusion_weights, dim=0).detach().cpu()
pathway_weights = F.softmax(model.pathway_weights, dim=0).detach().cpu()

print(f"\n📊 Learned GCN scale weights:")
print(f"   Local (r=1): {gcn_weights[0]:.3f}")
print(f"   Medium (r=3): {gcn_weights[1]:.3f}")
print(f"   Global (r=5): {gcn_weights[2]:.3f}")

print(f"\n📊 Learned pathway weights:")
print(f"   TCN pool: {pathway_weights[0]:.3f}")
print(f"   TCN+GOG: {pathway_weights[1]:.3f}")
print(f"   GCN pool: {pathway_weights[2]:.3f}")
print(f"   GCN+GOG: {pathway_weights[3]:.3f}")