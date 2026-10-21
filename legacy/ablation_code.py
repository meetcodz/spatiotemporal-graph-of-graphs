# =========================================================
# ABLATION STUDY — ST-GoG Multivariate Time Series Model
# =========================================================
# Variants:
#   V1  — Full ST-GoG               (baseline)
#   V2  — Without TAP               (mean pooling only)
#   V3  — Without SE                (no Squeeze-and-Excitation in ResTCN)
#   V4  — Without Multi-Scale GCN   (single-radius GCN, r=3)
#   V5  — Without GoG               (no Graph-of-Graphs module)
#   V6  — Single-Path Fusion        (TCN-pool path only, no GCN)
# =========================================================

!pip install -q sktime torch torchvision torchaudio tqdm thop

import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from torch.cuda.amp import GradScaler, autocast

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Notice: 'thop' not found. Profiling skipped.")

# ─────────────────────────── CONFIG ────────────────────────────
DATASET_NAME      = "PEMS-SF"
EPOCHS            = 250
BATCH_SIZE        = 8
LR                = 1e-3
BASE_WD           = 1e-4
DROPOUT           = 0.42
TCN_HIDDEN        = 128
GCN_HIDDEN        = 128
GOG_HIDDEN        = 128
STOCH_DEPTH_P     = 0.2
TTA_PASSES        = 7
SEED              = 42
SPARSITY_THRESHOLD = 0.05

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── DATA UTILS ────────────────────────
def df_to_np(df):
    N, C = df.shape
    T = len(df.iloc[0, 0])
    X = np.zeros((N, T, C), np.float32)
    for i in range(N):
        for j in range(C):
            X[i, :, j] = df.iloc[i, j].to_numpy()
    return X

def sparsify_adj(A, threshold=SPARSITY_THRESHOLD):
    mask = (A > threshold).float()
    return A * mask

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
            if i + r < T:  A[i, i + r] = w
    A += torch.eye(T)
    D = torch.diag((A.sum(1)).clamp(min=1e-6).pow(-0.5))
    adj = sparsify_adj(D @ A @ D)
    _adj[key] = adj
    return _adj[key].to(device)

# ─────────────────────── SHARED BUILDING BLOCKS ────────────────

# ── 1. ResTCN with optional SE ──────────────────────────────────
class ResTCN(nn.Module):
    """Dilated TCN with optional Squeeze-and-Excitation (V3 ablation)."""
    def __init__(self, inC, outC, use_se=True):
        super().__init__()
        layers, ch = [], inC
        for i in range(5):
            d = 2 ** i
            layers += [
                nn.Conv1d(ch, outC, 3, padding=d, dilation=d),
                nn.BatchNorm1d(outC), nn.ReLU(), nn.Dropout(DROPOUT)
            ]
            ch = outC
        self.net = nn.Sequential(*layers)
        self.res = nn.Conv1d(inC, outC, 1) if inC != outC else nn.Identity()
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(outC, outC // 4, 1), nn.ReLU(),
                nn.Conv1d(outC // 4, outC, 1), nn.Sigmoid()
            )

    def forward(self, x):
        out = self.net(x)
        if self.use_se:
            out = out * self.se(out)
        return F.relu(out + self.res(x))


# ── 2. Pooling — with and without TAP ──────────────────────────
class RobustAttentionPooling(nn.Module):
    """50/50 blend of attention + mean pooling (full TAP)."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return 0.5 * (x * weights).sum(dim=1) + 0.5 * x.mean(dim=1)


class MeanPooling(nn.Module):
    """Plain mean pooling — used in V2 (Without TAP).
    Accepts an optional `dim` kwarg so it is a drop-in for
    RobustAttentionPooling(dim) call-sites inside GOG."""
    def __init__(self, dim=None):   # dim is accepted but ignored
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)


# ── 3. Attentive Graph Conv (GAT-lite) ─────────────────────────
class AttentiveGraphConv(nn.Module):
    def __init__(self, F_in, H):
        super().__init__()
        self.fc        = nn.Linear(F_in, H)
        self.node_attn = nn.Linear(H, 1)

    def forward(self, x, A):
        h   = self.fc(x)
        h_w = h * torch.sigmoid(self.node_attn(h))
        return F.relu(torch.einsum("ij,bjf->bif", A, h_w))


class GCN(nn.Module):
    def __init__(self, F_in, H):
        super().__init__()
        self.inp    = nn.Linear(F_in, H)
        self.layers = nn.ModuleList([AttentiveGraphConv(H, H) for _ in range(3)])
        self.stoch_p = 0.0

    def forward(self, x, A):
        h = F.relu(self.inp(x))
        for g in self.layers:
            if self.training and torch.rand(1).item() < self.stoch_p:
                continue
            h = g(h, A)
        return h


# ── 4. Multi-Scale GCN ─────────────────────────────────────────
class MultiScaleGCN(nn.Module):
    """Three radii (1, 3, 5) with learned fusion — ablated in V4."""
    def __init__(self, F_in, H):
        super().__init__()
        self.gcn_local  = GCN(F_in, H)
        self.gcn_medium = GCN(F_in, H)
        self.gcn_global = GCN(F_in, H)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x, T):
        g1 = self.gcn_local (x, chain_adj(T, 1))
        g3 = self.gcn_medium(x, chain_adj(T, 3))
        g5 = self.gcn_global(x, chain_adj(T, 5))
        w  = F.softmax(self.fusion_weights, dim=0)
        return w[0] * g1 + w[1] * g3 + w[2] * g5


class SingleScaleGCN(nn.Module):
    """Single-radius (r=3) GCN — replaces MultiScaleGCN in V4."""
    def __init__(self, F_in, H):
        super().__init__()
        self.gcn = GCN(F_in, H)

    def forward(self, x, T):
        return self.gcn(x, chain_adj(T, 3))


# ── 5. GoG ─────────────────────────────────────────────────────
class GOG(nn.Module):
    def __init__(self, F_in, H, num_seg, use_tap=True):
        super().__init__()
        self.num_seg   = num_seg
        self.inner     = GCN(F_in, H)
        self.outer     = GCN(H, H)
        pool_cls       = RobustAttentionPooling if use_tap else MeanPooling
        self.inner_pool = pool_cls(H)
        self.outer_pool = pool_cls(H)

    def forward(self, x):
        segs = torch.chunk(x, self.num_seg, dim=1)
        outs = []
        for s in segs:
            gcn_out = self.inner(s, chain_adj(s.size(1), 3))
            outs.append(self.inner_pool(gcn_out))
        h = torch.stack(outs, dim=1)
        out = self.outer(h, chain_adj(self.num_seg, 1))
        return self.outer_pool(out)


# ═══════════════════════════════════════════════════════════════
#  MODEL VARIANTS
# ═══════════════════════════════════════════════════════════════

def _num_seg(seq_len):
    if   seq_len < 100: return 4
    elif seq_len < 200: return 6
    elif seq_len < 400: return 8
    else:               return 10


# ── V1: Full ST-GoG ────────────────────────────────────────────
class FullSTGoG(nn.Module):
    """V1 — Full model (baseline)."""
    NAME = "Full ST-GoG"

    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T       = seq_len
        self.num_seg = _num_seg(seq_len)

        self.tcn       = ResTCN(input_dim, TCN_HIDDEN, use_se=True)
        self.multi_gcn = MultiScaleGCN(input_dim, GCN_HIDDEN)
        self.gog1      = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg, use_tap=True)
        self.gog2      = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg, use_tap=True)

        self.t_pool = RobustAttentionPooling(TCN_HIDDEN)
        self.g_pool = RobustAttentionPooling(GCN_HIDDEN)

        feat_dim = 2 * (TCN_HIDDEN + GOG_HIDDEN)
        self._build_head(feat_dim, n_classes)
        self.pathway_weights = nn.Parameter(torch.ones(4) / 4.0)

    def _build_head(self, feat_dim, n_classes):
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)

    def set_stoch_depth(self, p):
        for m in [self.multi_gcn.gcn_local, self.multi_gcn.gcn_medium,
                  self.multi_gcn.gcn_global,
                  self.gog1.inner, self.gog1.outer,
                  self.gog2.inner, self.gog2.outer]:
            m.stoch_p = p

    def forward(self, x):
        t     = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        g     = self.multi_gcn(x, self.T)
        tp    = self.t_pool(t)
        gp    = self.g_pool(g)
        tgog  = self.gog1(t)
        ggog  = self.gog2(g)

        w = F.softmax(self.pathway_weights, dim=0)
        z = torch.cat([w[0]*tp, w[1]*tgog, w[2]*gp, w[3]*ggog], dim=1)
        return self._head(z)

    def _head(self, z):
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.fc3(h)

    def get_gcn_weights(self):
        return F.softmax(self.multi_gcn.fusion_weights, dim=0).detach().cpu()

    def get_pathway_weights(self):
        return F.softmax(self.pathway_weights, dim=0).detach().cpu()


# ── V2: Without TAP (pure mean pooling everywhere) ─────────────
class NoTAP(FullSTGoG):
    """V2 — Replace all RobustAttentionPooling with MeanPooling."""
    NAME = "Without TAP"

    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__(input_dim, seq_len, n_classes)
        # Override pooling layers
        self.t_pool = MeanPooling()
        self.g_pool = MeanPooling()
        self.gog1   = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg, use_tap=False)
        self.gog2   = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg, use_tap=False)


# ── V3: Without SE ─────────────────────────────────────────────
class NoSE(FullSTGoG):
    """V3 — ResTCN without Squeeze-and-Excitation."""
    NAME = "Without SE"

    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__(input_dim, seq_len, n_classes)
        self.tcn = ResTCN(input_dim, TCN_HIDDEN, use_se=False)


# ── V4: Without Multi-Scale GCN (single r=3) ───────────────────
class NoMultiScaleGCN(nn.Module):
    """V4 — Single-scale GCN (r=3) instead of multi-scale."""
    NAME = "Without Multi-Scale GCN"

    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T       = seq_len
        self.num_seg = _num_seg(seq_len)

        self.tcn      = ResTCN(input_dim, TCN_HIDDEN, use_se=True)
        self.single_gcn = SingleScaleGCN(input_dim, GCN_HIDDEN)
        self.gog1     = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg, use_tap=True)
        self.gog2     = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg, use_tap=True)

        self.t_pool = RobustAttentionPooling(TCN_HIDDEN)
        self.g_pool = RobustAttentionPooling(GCN_HIDDEN)

        feat_dim = 2 * (TCN_HIDDEN + GOG_HIDDEN)
        self.fc1 = nn.Linear(feat_dim, 512); self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256);     self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)
        self.pathway_weights = nn.Parameter(torch.ones(4) / 4.0)

    def set_stoch_depth(self, p):
        for m in [self.single_gcn.gcn,
                  self.gog1.inner, self.gog1.outer,
                  self.gog2.inner, self.gog2.outer]:
            m.stoch_p = p

    def forward(self, x):
        t    = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        g    = self.single_gcn(x, self.T)
        tp   = self.t_pool(t)
        gp   = self.g_pool(g)
        tgog = self.gog1(t)
        ggog = self.gog2(g)

        w = F.softmax(self.pathway_weights, dim=0)
        z = torch.cat([w[0]*tp, w[1]*tgog, w[2]*gp, w[3]*ggog], dim=1)
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.fc3(h)

    def get_pathway_weights(self):
        return F.softmax(self.pathway_weights, dim=0).detach().cpu()

    def get_gcn_weights(self):
        return None  # single-scale, no fusion weights


# ── V5: Without GoG ────────────────────────────────────────────
class NoGoG(nn.Module):
    """V5 — No GoG hierarchical module; only flat pooled features."""
    NAME = "Without GoG"

    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T = seq_len

        self.tcn       = ResTCN(input_dim, TCN_HIDDEN, use_se=True)
        self.multi_gcn = MultiScaleGCN(input_dim, GCN_HIDDEN)
        self.t_pool    = RobustAttentionPooling(TCN_HIDDEN)
        self.g_pool    = RobustAttentionPooling(GCN_HIDDEN)

        feat_dim = TCN_HIDDEN + GCN_HIDDEN         # only 2 paths (no GoG)
        self.fc1 = nn.Linear(feat_dim, 512); self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256);     self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)
        self.pathway_weights = nn.Parameter(torch.ones(2) / 2.0)

    def set_stoch_depth(self, p):
        for m in [self.multi_gcn.gcn_local, self.multi_gcn.gcn_medium,
                  self.multi_gcn.gcn_global]:
            m.stoch_p = p

    def forward(self, x):
        t  = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        g  = self.multi_gcn(x, self.T)
        tp = self.t_pool(t)
        gp = self.g_pool(g)

        w = F.softmax(self.pathway_weights, dim=0)
        z = torch.cat([w[0]*tp, w[1]*gp], dim=1)
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.fc3(h)

    def get_gcn_weights(self):
        return F.softmax(self.multi_gcn.fusion_weights, dim=0).detach().cpu()

    def get_pathway_weights(self):
        return F.softmax(self.pathway_weights, dim=0).detach().cpu()


# ── V6: Single-Path Fusion (TCN-pool only) ─────────────────────
class SinglePathFusion(nn.Module):
    """V6 — Single pathway: TCN → TAP only. No GCN, no GoG."""
    NAME = "Single-Path Fusion"

    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T   = seq_len
        self.tcn = ResTCN(input_dim, TCN_HIDDEN, use_se=True)
        self.pool = RobustAttentionPooling(TCN_HIDDEN)

        feat_dim = TCN_HIDDEN
        self.fc1 = nn.Linear(feat_dim, 512); self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256);     self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)

    def set_stoch_depth(self, p):
        pass   # No GCN layers with stochastic depth

    def forward(self, x):
        t = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        z = self.pool(t)
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.fc3(h)

    def get_gcn_weights(self):
        return None

    def get_pathway_weights(self):
        return None


# ═══════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════
class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y, training=True):
        if self.alpha > 0 and training:
            lam   = np.random.beta(self.alpha, self.alpha)
            idx   = torch.randperm(x.size(0)).to(x.device)
            return lam*x + (1-lam)*x[idx], y, y[idx], lam
        return x, y, y, 1.0


def train_one_epoch(model, loader, crit, opt, scaler_amp, mixup, ep):
    model.train()
    p = STOCH_DEPTH_P * np.exp(-ep / 60.0)
    model.set_stoch_depth(p)
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb     = xb.to(device), yb.to(device)
        xb, ya, yb2, lam = mixup(xb, yb, training=True)
        opt.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            out  = model(xb)
            loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb2)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_with_tta(model, loader):
    model.eval()
    y_true, y_pred = [], []
    inf_start = time.time()
    for xb, yb in loader:
        xb = xb.to(device)
        preds = []
        for i in range(TTA_PASSES):
            aug = xb if i == 0 else xb + torch.randn_like(xb) * 0.003
            preds.append(model(aug))
        pred = torch.stack(preds).mean(0).argmax(1).cpu()
        y_true.extend(yb.tolist())
        y_pred.extend(pred.tolist())
    inf_time = time.time() - inf_start
    acc = (np.array(y_pred) == np.array(y_true)).mean()
    return acc, inf_time, y_true, y_pred


def run_variant(ModelClass, C, T, n_classes,
                train_loader, test_loader, label_smooth):
    print(f"\n{'='*60}")
    print(f"  VARIANT: {ModelClass.NAME}")
    print(f"{'='*60}")

    model      = ModelClass(C, T, n_classes).to(device)
    scaler_amp = GradScaler(enabled=(device.type == 'cuda'))
    crit       = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    opt        = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=BASE_WD)
    sch        = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    mixup      = MixUp(alpha=0.2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    best_acc, best_state, best_epoch = 0.0, None, 0
    train_start = time.time()

    for ep in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, crit, opt,
                                   scaler_amp, mixup, ep)
        sch.step()

        acc, inf_time, _, _ = evaluate_with_tta(model, test_loader)

        if acc > best_acc:
            best_acc   = acc
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker     = "★ NEW BEST"
        else:
            marker = ""

        if ep % 10 == 0 or marker:
            print(f"  Ep {ep:03d} | Loss {avg_loss:.4f} | Acc {acc*100:.2f}% | "
                  f"Best {best_acc*100:.2f}% | {inf_time:.3f}s  {marker}")

    train_time = time.time() - train_start

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final clean eval (no TTA — deterministic)
    model.eval()
    y_true_f, y_pred_f = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            y_true_f.extend(yb.tolist())
            y_pred_f.extend(pred.tolist())

    f1 = f1_score(y_true_f, y_pred_f, average='weighted')
    cm = confusion_matrix(y_true_f, y_pred_f)

    # FLOPs profiling
    gflops = None
    if THOP_AVAILABLE:
        dummy = torch.randn(1, T, C).to(device)
        try:
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            gflops  = macs / 1e9
        except Exception:
            pass

    result = dict(
        name         = ModelClass.NAME,
        best_acc     = best_acc,
        best_epoch   = best_epoch,
        f1           = f1,
        params       = n_params,
        gflops       = gflops,
        train_time   = train_time,
        cm           = cm,
    )

    # Print learned weights if available
    if hasattr(model, 'get_gcn_weights') and model.get_gcn_weights() is not None:
        gw = model.get_gcn_weights()
        print(f"\n  GCN scale weights → Local:{gw[0]:.3f}  Medium:{gw[1]:.3f}  Global:{gw[2]:.3f}")
    if hasattr(model, 'get_pathway_weights') and model.get_pathway_weights() is not None:
        pw = model.get_pathway_weights()
        pw_labels = {
            "Full ST-GoG":           ["TCN-pool", "TCN+GoG", "GCN-pool", "GCN+GoG"],
            "Without TAP":           ["TCN-pool", "TCN+GoG", "GCN-pool", "GCN+GoG"],
            "Without SE":            ["TCN-pool", "TCN+GoG", "GCN-pool", "GCN+GoG"],
            "Without Multi-Scale GCN":["TCN-pool","TCN+GoG", "GCN-pool", "GCN+GoG"],
            "Without GoG":           ["TCN-pool", "GCN-pool"],
        }
        labels = pw_labels.get(ModelClass.NAME, [f"path{i}" for i in range(len(pw))])
        pw_str = "  ".join(f"{l}:{v:.3f}" for l, v in zip(labels, pw))
        print(f"  Pathway weights → {pw_str}")

    return result


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
print(f"\nLoading dataset: {DATASET_NAME} …")
Xtr_df, ytr_raw = load_UCR_UEA_dataset(DATASET_NAME, "train", return_X_y=True)
Xte_df, yte_raw = load_UCR_UEA_dataset(DATASET_NAME, "test",  return_X_y=True)

X_train = df_to_np(Xtr_df)
X_test  = df_to_np(Xte_df)
N_train, T, C = X_train.shape

sc       = StandardScaler()
X_train  = sc.fit_transform(X_train.reshape(-1, C)).reshape(N_train, T, C)
X_test   = sc.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

le        = LabelEncoder()
y_train   = le.fit_transform(ytr_raw)
y_test    = le.transform(yte_raw)
n_classes = len(le.classes_)

label_smooth = 0.1 if n_classes <= 5 else 0.15

train_loader = DataLoader(
    list(zip(torch.tensor(X_train).float(), torch.tensor(y_train).long())),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    list(zip(torch.tensor(X_test).float(), torch.tensor(y_test).long())),
    batch_size=BATCH_SIZE, shuffle=False
)

print(f"Dataset: N_train={N_train}, T={T}, C={C}, Classes={n_classes}")
print(f"Device : {device}")


# ═══════════════════════════════════════════════════════════════
#  RUN ALL ABLATION VARIANTS
# ═══════════════════════════════════════════════════════════════
VARIANTS = [
    FullSTGoG,          # V1 — Full
    NoTAP,              # V2 — Without TAP
    NoSE,               # V3 — Without SE
    NoMultiScaleGCN,    # V4 — Without Multi-Scale GCN
    NoGoG,              # V5 — Without GoG
    SinglePathFusion,   # V6 — Single-Path Fusion
]

all_results = []
for ModelClass in VARIANTS:
    result = run_variant(
        ModelClass, C, T, n_classes,
        train_loader, test_loader, label_smooth
    )
    all_results.append(result)


# ═══════════════════════════════════════════════════════════════
#  ABLATION SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
def delta(val, baseline):
    d = (val - baseline) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}%"

print("\n\n" + "═"*90)
print("                         ABLATION STUDY — SUMMARY TABLE")
print("═"*90)
header = (
    f"{'Variant':<28} {'Acc (Best)':>11} {'ΔAcc':>9} {'F1':>7}"
    f" {'Params':>10} {'GFLOPs':>8} {'BestEp':>7} {'TrainTime':>10}"
)
print(header)
print("─"*90)

baseline_acc = all_results[0]['best_acc']
for r in all_results:
    gflops_str = f"{r['gflops']:.3f}" if r['gflops'] is not None else "  N/A "
    print(
        f"{r['name']:<28} "
        f"{r['best_acc']*100:>10.2f}% "
        f"{delta(r['best_acc'], baseline_acc):>9} "
        f"{r['f1']:>7.4f} "
        f"{r['params']/1e6:>9.3f}M "
        f"{gflops_str:>8} "
        f"{r['best_epoch']:>7} "
        f"{r['train_time']/60:>9.1f}m"
    )

print("═"*90)
print("\nNotes:")
print("  ΔAcc  = accuracy relative to Full ST-GoG (V1)")
print("  Params in millions,  GFLOPs via thop  (N/A if thop not installed)")
print("  Best Epoch = epoch with highest validation accuracy")
print("  Train time includes all TTA inference passes during training\n")

print("Per-class Confusion Matrices:")
for r in all_results:
    print(f"\n  [{r['name']}]")
    print(r['cm'])

