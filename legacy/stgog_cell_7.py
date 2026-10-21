# ===========================================================================
# Multi-GPU Time-Series Classifier — Kaggle 2× T4
# Includes: Snapshot Ensembling + Full Model Evaluation
# ===========================================================================

!pip install -q sktime torch torchvision torchaudio

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_NAME  = "SelfRegulationSCP1"
EPOCHS        = 250
BATCH_SIZE    = 16          # 8 per GPU × 2 GPUs
LR            = 1e-3
BASE_WD       = 1e-4
DROPOUT       = 0.42
TCN_HIDDEN    = 128
GCN_HIDDEN    = 128
GOG_HIDDEN    = 128
STOCH_DEPTH_P = 0.2
TTA_PASSES    = 7
SEED          = 42
OUT_DIR       = "/kaggle/working"   # change to "." if running locally

os.makedirs(OUT_DIR, exist_ok=True)

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── MULTI-GPU SETUP ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
USE_AMP = device.type == "cuda"

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print(f"  Device : {device}")
print(f"  GPUs   : {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"    GPU {i} — {props.name}  "
          f"({props.total_memory // 1024**2} MB VRAM)")
print("=" * 60)

amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# helper: reach through DataParallel wrapper if present
def base(m):
    return m.module if isinstance(m, nn.DataParallel) else m

# ── DATA UTILS ───────────────────────────────────────────────────────────────
def df_to_np(df):
    N, C = df.shape
    T = len(df.iloc[0, 0])
    X = np.zeros((N, T, C), np.float32)
    for i in range(N):
        for j in range(C):
            X[i, :, j] = df.iloc[i, j].to_numpy()
    return X

_adj_cache = {}
def chain_adj(T, radius, dev):
    """Symmetric normalised adjacency for a chain graph (cached per device)."""
    key = (T, radius, str(dev))
    if key in _adj_cache:
        return _adj_cache[key]
    A = torch.zeros(T, T)
    for i in range(T):
        for r in range(1, radius + 1):
            w = 1.0 / r
            if i - r >= 0: A[i, i - r] = w
            if i + r < T:  A[i, i + r] = w
    A += torch.eye(T)
    D = A.sum(1).clamp(min=1e-6).pow(-0.5).diag()
    _adj_cache[key] = (D @ A @ D).to(dev)
    return _adj_cache[key]

# ── MODEL COMPONENTS ─────────────────────────────────────────────────────────
class ResTCN(nn.Module):
    """Dilated residual TCN with Squeeze-Excitation channel attention."""
    def __init__(self, inC, outC):
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
        self.se  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(outC, max(1, outC // 4), 1), nn.ReLU(),
            nn.Conv1d(max(1, outC // 4), outC, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return F.relu(out * self.se(out) + self.res(x))


class GraphConv(nn.Module):
    def __init__(self, F_in, H):
        super().__init__()
        self.fc = nn.Linear(F_in, H)

    def forward(self, x, A):
        return F.relu(self.fc(torch.einsum("ij,bjf->bif", A, x)))


class GCN(nn.Module):
    def __init__(self, F_in, H):
        super().__init__()
        self.inp    = nn.Linear(F_in, H)
        self.layers = nn.ModuleList([GraphConv(H, H) for _ in range(3)])
        self.stoch_p = 0.0

    def forward(self, x, A):
        h = F.relu(self.inp(x))
        for g in self.layers:
            if self.training and torch.rand(1).item() < self.stoch_p:
                continue
            h = g(h, A)
        return h


class GOG(nn.Module):
    """Graph-of-Graphs: inner GCN per segment → outer GCN across segments."""
    def __init__(self, F_in, H, num_seg):
        super().__init__()
        self.num_seg = num_seg
        self.inner   = GCN(F_in, H)
        self.outer   = GCN(H, H)

    def forward(self, x):
        dev  = x.device
        outs = []
        for s in torch.chunk(x, self.num_seg, dim=1):
            A = chain_adj(s.size(1), 3, dev)
            outs.append(self.inner(s, A).mean(1))
        h = torch.stack(outs, dim=1)
        return self.outer(h, chain_adj(self.num_seg, 1, dev)).mean(1)


class MultiScaleGCN(nn.Module):
    """Three GCNs at different radii with learnable softmax fusion."""
    def __init__(self, F_in, H):
        super().__init__()
        self.gcn_local  = GCN(F_in, H)
        self.gcn_medium = GCN(F_in, H)
        self.gcn_global = GCN(F_in, H)
        self.fusion_w   = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x, T):
        dev = x.device
        g1  = self.gcn_local (x, chain_adj(T, 1, dev))
        g3  = self.gcn_medium(x, chain_adj(T, 3, dev))
        g5  = self.gcn_global(x, chain_adj(T, 5, dev))
        w   = F.softmax(self.fusion_w, dim=0)
        return w[0] * g1 + w[1] * g3 + w[2] * g5


class Model(nn.Module):
    def __init__(self, input_dim, seq_len, n_classes):
        super().__init__()
        self.T = seq_len
        self.C = input_dim
        self.num_seg = (4 if seq_len < 100 else
                        6 if seq_len < 200 else
                        8 if seq_len < 400 else 10)

        self.tcn       = ResTCN(input_dim, TCN_HIDDEN)
        self.multi_gcn = MultiScaleGCN(input_dim, GCN_HIDDEN)
        self.gog1      = GOG(TCN_HIDDEN, GOG_HIDDEN, self.num_seg)
        self.gog2      = GOG(GCN_HIDDEN, GOG_HIDDEN, self.num_seg)

        feat_dim = 2 * (TCN_HIDDEN + GOG_HIDDEN)
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)

        self.pathway_w = nn.Parameter(torch.ones(4) / 4.0)

    def set_stoch_depth(self, p):
        for m in [self.multi_gcn.gcn_local, self.multi_gcn.gcn_medium,
                  self.multi_gcn.gcn_global,
                  self.gog1.inner, self.gog1.outer,
                  self.gog2.inner, self.gog2.outer]:
            m.stoch_p = p

    def forward(self, x):
        t = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        g = self.multi_gcn(x, self.T)

        t_pool = t.mean(1)
        g_pool = g.mean(1)
        t_gog  = self.gog1(t)
        g_gog  = self.gog2(g)

        w = F.softmax(self.pathway_w, dim=0)
        z = torch.cat([w[0]*t_pool, w[1]*t_gog,
                       w[2]*g_pool, w[3]*g_gog], dim=1)

        h = F.dropout(F.relu(self.bn1(self.fc1(z))), p=DROPOUT, training=self.training)
        h = F.dropout(F.relu(self.bn2(self.fc2(h))), p=DROPOUT, training=self.training)
        return self.fc3(h)


class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        lam   = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idx   = torch.randperm(x.size(0), device=x.device)
        mixed = lam * x + (1 - lam) * x[idx]
        return mixed, y, y[idx], lam


# ── DATA LOADING ─────────────────────────────────────────────────────────────
print(f"\nLoading {DATASET_NAME}...")
Xtr_df, ytr_raw = load_UCR_UEA_dataset(DATASET_NAME, "train", return_X_y=True)
Xte_df, yte_raw = load_UCR_UEA_dataset(DATASET_NAME, "test",  return_X_y=True)

X_train = df_to_np(Xtr_df)
X_test  = df_to_np(Xte_df)
N_tr, T, C = X_train.shape

sc = StandardScaler()
X_train = sc.fit_transform(X_train.reshape(-1, C)).reshape(N_tr, T, C)
X_test  = sc.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

le       = LabelEncoder()
y_train  = le.fit_transform(ytr_raw)
y_test   = le.transform(yte_raw)
n_classes = len(le.classes_)

# pin_memory=True speeds up CPU→GPU transfer on Kaggle
train_ds = TensorDataset(torch.tensor(X_train).float(),
                         torch.tensor(y_train).long())
test_ds  = TensorDataset(torch.tensor(X_test).float(),
                         torch.tensor(y_test).long())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"  Train: {N_tr} samples  |  Test: {len(y_test)} samples")
print(f"  Seq len T={T}, Channels C={C}, Classes={n_classes}")

# ── MODEL & OPTIMISER SETUP ──────────────────────────────────────────────────
_model = Model(C, T, n_classes).to(device)

if n_gpus > 1:
    model = nn.DataParallel(_model, device_ids=list(range(n_gpus)))
    print(f"\nDataParallel enabled across {n_gpus} GPUs")
else:
    model = _model

label_smooth = 0.1 if n_classes <= 5 else 0.15
crit  = nn.CrossEntropyLoss(label_smoothing=label_smooth)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=BASE_WD)
sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
mixup = MixUp(alpha=0.2)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {n_params:,}")
print(f"  Segments   : {base(model).num_seg}")
print(f"  Batch size : {BATCH_SIZE}  ({BATCH_SIZE//max(n_gpus,1)} per GPU)")

# ── SNAPSHOT EPOCH SCHEDULE ──────────────────────────────────────────────────
# CosineAnnealingWarmRestarts T_0=50 T_mult=2 → restarts at 50, 150
SNAPSHOT_EPOCHS = set()
_t0, _ep = 50, 0
while True:
    _ep += _t0
    if _ep > EPOCHS: break
    SNAPSHOT_EPOCHS.add(_ep)
    _t0 *= 2
print(f"  Snapshot saves at epochs : {sorted(SNAPSHOT_EPOCHS)}\n")

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
best_acc   = 0.0
best_state = None
best_epoch = 0
snapshots  = []       # (epoch, state_dict)
t_start    = time.time()

for ep in range(1, EPOCHS + 1):
    model.train()
    current_stoch_p = STOCH_DEPTH_P * np.exp(-ep / 60.0)
    base(model).set_stoch_depth(current_stoch_p)

    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        xb_m, ya, yb_m, lam = mixup(xb, yb)

        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out  = model(xb_m)
            loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb_m)

        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp_scaler.step(opt)
        amp_scaler.update()
        total_loss += loss.item()

    sch.step()

    # ── Snapshot at cosine restart minima ────────────────────────────────
    if ep in SNAPSHOT_EPOCHS:
        snap = {k: v.cpu().clone()
                for k, v in base(model).state_dict().items()}
        snapshots.append(snap)
        print(f"  📸  Snapshot saved — epoch {ep}")

    # ── Evaluation ───────────────────────────────────────────────────────
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            preds = [model(xb)]
            for _ in range(TTA_PASSES - 1):
                preds.append(model(xb + torch.randn_like(xb) * 0.003))
            pred = torch.stack(preds).mean(0).argmax(1).cpu()
            correct += (pred == yb).sum().item()

    acc     = correct / len(y_test)
    avg_loss = total_loss / len(train_loader)
    elapsed  = (time.time() - t_start) / 60

    if acc > best_acc:
        best_acc   = acc
        best_epoch = ep
        best_state = {k: v.cpu().clone()
                      for k, v in base(model).state_dict().items()}
        marker = "★ NEW BEST"
    else:
        marker = ""

    if ep % 10 == 0 or marker:
        lr_now = opt.param_groups[0]["lr"]
        print(f"Ep {ep:03d} | Loss {avg_loss:.4f} | Acc {acc*100:.2f}% | "
              f"Best {best_acc*100:.2f}% | LR {lr_now:.2e} | "
              f"{elapsed:.1f}m  {marker}")

total_time = (time.time() - t_start) / 60
print(f"\n{'='*60}")
print(f"Training done in {total_time:.1f} min")
print(f"Best single-model acc : {best_acc*100:.2f}%  (epoch {best_epoch})")
print(f"{'='*60}")

# ── SNAPSHOT ENSEMBLE PREDICTION ─────────────────────────────────────────────
def ensemble_predict(loader, state_dicts, ref_model, device, tta=TTA_PASSES):
    """
    Average softmax logits across all snapshots.
    Each snapshot gets its own TTA passes for maximum diversity.
    """
    all_logits = []
    for state in state_dicts:
        base(ref_model).load_state_dict(state)
        ref_model.eval()
        batch_logits = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                passes = [ref_model(xb)]
                for _ in range(tta - 1):
                    passes.append(ref_model(xb + torch.randn_like(xb) * 0.003))
                batch_logits.append(torch.stack(passes).mean(0).cpu())
        all_logits.append(torch.cat(batch_logits, dim=0))
    return torch.stack(all_logits).mean(0)    # (N, n_classes)

all_states = snapshots + [best_state]
print(f"\nRunning snapshot ensemble  "
      f"({len(snapshots)} snapshots + best checkpoint = {len(all_states)} models)")

ensemble_logits = ensemble_predict(test_loader, all_states, model, device)
ens_preds       = ensemble_logits.argmax(1).numpy()
ens_acc         = (ens_preds == y_test).mean()
delta           = ens_acc - best_acc

print(f"\n{'='*60}")
print(f"  Single best model  : {best_acc*100:.2f}%")
print(f"  Snapshot ensemble  : {ens_acc*100:.2f}%  "
      f"({'↑' if delta > 0 else '→'}{abs(delta)*100:.2f}% lift)")
print(f"{'='*60}")

# ── FULL MODEL EVALUATION ────────────────────────────────────────────────────
print("\n── Classification Report ──")
print(classification_report(y_test, ens_preds,
                             target_names=[str(c) for c in le.classes_],
                             digits=3))

# ── Calibration (ECE) ────────────────────────────────────────────────────────
probs       = torch.softmax(ensemble_logits, -1).numpy()
confidences = probs.max(axis=1)
correctness = (ens_preds == y_test).astype(float)
n_bins      = 10
bin_edges   = np.linspace(0, 1, n_bins + 1)
bin_accs    = np.zeros(n_bins)
bin_confs   = np.zeros(n_bins)
bin_sizes   = np.zeros(n_bins)

for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    m = (confidences >= lo) & (confidences < hi)
    if m.sum() > 0:
        bin_accs[i]  = correctness[m].mean()
        bin_confs[i] = confidences[m].mean()
        bin_sizes[i] = m.sum()

ece = (np.abs(bin_accs - bin_confs) * bin_sizes / len(y_test)).sum()
print(f"ECE : {ece:.4f}  (lower = better calibrated; <0.05 is good)")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
class_names = [str(c) for c in le.classes_]
cm_norm     = confusion_matrix(y_test, ens_preds, normalize="true")
per_cls_acc = cm_norm.diagonal()

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ── 1. Confusion matrix ───────────────────────────────────────────────────────
ax1  = fig.add_subplot(gs[0, :2])
disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
disp.plot(ax=ax1, colorbar=True, cmap="Blues", values_format=".2f")
ax1.set_title("Confusion matrix (row-normalised)", fontsize=12)
ax1.tick_params(axis="x", rotation=45)

# ── 2. Per-class accuracy bar chart ───────────────────────────────────────────
ax2    = fig.add_subplot(gs[0, 2])
colors = ["#3B8BD4" if a >= 0.85 else "#EF9F27" if a >= 0.65 else "#E24B4A"
          for a in per_cls_acc]
bars   = ax2.bar(range(len(class_names)), per_cls_acc * 100, color=colors,
                  edgecolor="white", linewidth=0.5)
ax2.axhline(ens_acc * 100, color="gray", linestyle="--", linewidth=1,
            label=f"Overall {ens_acc*100:.1f}%")
ax2.set_xticks(range(len(class_names)))
ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 110)
ax2.set_title("Per-class accuracy", fontsize=12)
ax2.legend(fontsize=9)
for bar, a in zip(bars, per_cls_acc):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{a*100:.0f}%", ha="center", va="bottom", fontsize=8)

# ── 3. Calibration / ECE plot ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
gap = bin_accs - bin_confs
ax3.bar(bin_edges[:-1], bin_confs, width=1/n_bins, align="edge",
        color="#378ADD", alpha=0.65, label="Mean confidence")
ax3.bar(bin_edges[:-1], bin_accs,  width=1/n_bins, align="edge",
        color="none", edgecolor="#185FA5", linewidth=1.4, label="Accuracy")
ax3.bar(bin_edges[:-1], np.where(gap < 0, -gap, 0), width=1/n_bins,
        align="edge", bottom=bin_confs + np.where(gap < 0, gap, 0),
        color="#E24B4A", alpha=0.45, label="Underconfident")
ax3.bar(bin_edges[:-1], np.where(gap > 0, gap, 0), width=1/n_bins,
        align="edge", bottom=bin_accs - np.where(gap > 0, gap, 0),
        color="#EF9F27", alpha=0.45, label="Overconfident")
ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4,
         label="Perfect calibration")
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_xlabel("Confidence"); ax3.set_ylabel("Accuracy")
ax3.set_title(f"Calibration  (ECE = {ece:.4f})", fontsize=12)
ax3.legend(fontsize=8, loc="upper left")

# ── 4. Confidence distribution ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(confidences[correctness == 1], bins=25, alpha=0.65,
         color="#3B8BD4", label="Correct")
ax4.hist(confidences[correctness == 0], bins=25, alpha=0.65,
         color="#E24B4A", label="Wrong")
ax4.axvline(confidences.mean(), color="gray", linestyle="--",
            linewidth=1, label=f"Mean conf {confidences.mean():.2f}")
ax4.set_xlabel("Max softmax confidence")
ax4.set_ylabel("Count")
ax4.set_title("Confidence distribution", fontsize=12)
ax4.legend(fontsize=9)

# ── 5. Training summary text panel ───────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
summary = (
    f"Dataset        {DATASET_NAME}\n"
    f"Seq len / Ch   T={T}  C={C}\n"
    f"Classes        {n_classes}\n"
    f"Train / Test   {N_tr} / {len(y_test)}\n\n"
    f"GPUs used      {n_gpus}× T4\n"
    f"Parameters     {n_params:,}\n"
    f"Epochs         {EPOCHS}\n"
    f"Batch size     {BATCH_SIZE}\n"
    f"Train time     {total_time:.1f} min\n\n"
    f"Snapshot models  {len(all_states)}\n"
    f"Single-model acc {best_acc*100:.2f}%\n"
    f"Ensemble acc     {ens_acc*100:.2f}%\n"
    f"ECE              {ece:.4f}"
)
ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
         fontsize=10, verticalalignment="top",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                   edgecolor="#cccccc", linewidth=0.8))
ax5.set_title("Run summary", fontsize=12)

fig.suptitle(
    f"{DATASET_NAME} — Snapshot Ensemble ({len(all_states)} models)  |  "
    f"Acc {ens_acc*100:.2f}%  |  ECE {ece:.4f}",
    fontsize=13, y=1.01
)

out_path = os.path.join(OUT_DIR, "model_evaluation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nEvaluation plot saved → {out_path}")

# ── FINAL LEARNED WEIGHT SUMMARY ─────────────────────────────────────────────
base(model).load_state_dict(best_state)
gcn_w  = F.softmax(base(model).multi_gcn.fusion_w, dim=0).detach().cpu()
path_w = F.softmax(base(model).pathway_w, dim=0).detach().cpu()

print(f"\nLearned GCN scale weights:")
print(f"  Local  (r=1)  {gcn_w[0]:.3f}")
print(f"  Medium (r=3)  {gcn_w[1]:.3f}")
print(f"  Global (r=5)  {gcn_w[2]:.3f}")
print(f"\nLearned pathway weights:")
print(f"  TCN pool      {path_w[0]:.3f}")
print(f"  TCN + GOG     {path_w[1]:.3f}")
print(f"  GCN pool      {path_w[2]:.3f}")
print(f"  GCN + GOG     {path_w[3]:.3f}")

print(f"\n{'='*60}")
print(f"  FINAL ENSEMBLE ACCURACY : {ens_acc*100:.2f}%")
print(f"{'='*60}")