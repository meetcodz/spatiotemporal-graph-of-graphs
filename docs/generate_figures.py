import matplotlib.pyplot as plt
import numpy as np

# Set overall aesthetic
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------
# Figure 1: Benchmark Comparison across Categories & Overall
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

categories = ['Motion & Activity\n(8 datasets)', 'Medical & Neuro\n(9 datasets)', 'Sensor & High-Dim\n(3 datasets)', 'Astronomy & Bio\n(2 datasets)', 'Overall Average\n(22 datasets)']
st_gog = [0.865, 0.697, 0.637, 0.626, 0.743]
sagog = [0.840, 0.696, 0.648, 0.742, 0.740]
rocket = [0.803, 0.621, 0.534, 0.697, 0.657]
simple_par = [0.776, 0.638, 0.632, 0.587, 0.697]

x = np.arange(len(categories))
width = 0.20

rects1 = ax.bar(x - 1.5*width, [v*100 for v in st_gog], width, label='ST-GoG (Proposed)', color='#1E3A8A', edgecolor='none')
rects2 = ax.bar(x - 0.5*width, [v*100 for v in sagog], width, label='SAGoG (Baseline)', color='#0D9488', edgecolor='none')
rects3 = ax.bar(x + 0.5*width, [v*100 for v in rocket], width, label='ROCKET (Kernels)', color='#F59E0B', edgecolor='none')
rects4 = ax.bar(x + 1.5*width, [v*100 for v in simple_par], width, label='Simple Parallel', color='#94A3B8', edgecolor='none')

ax.set_ylabel('Mean Classification Accuracy (%)', fontsize=11, fontweight='bold', color='#1E293B')
ax.set_title('MTSC Benchmark Performance by Domain Category (22 UEA Datasets)', fontsize=12, fontweight='bold', pad=15, color='#0F172A')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9.5, color='#334155')
ax.set_ylim(40, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=9, loc='upper right')

# Value annotations on ST-GoG overall
ax.annotate(f'74.3%', xy=(x[-1] - 1.5*width, 74.3), xytext=(0, 3),
            textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E3A8A')

plt.tight_layout()
fig.savefig('d:/first intern(maybe)/fig_benchmark_comparison.png', dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# Figure 2: Ablation Study Impact on PEMS-SF
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=300)

variants = [
    'Full ST-GoG',
    'w/o Temporal Attn (Mean Pool)',
    'w/o SE Attention',
    'w/o Multi-Scale GCN',
    'w/o GoG Module (Mean Only)'
]
accuracies = [87.28, 87.86, 86.71, 86.71, 84.39]
colors_list = ['#1E3A8A', '#0D9488', '#F59E0B', '#EA580C', '#DC2626']

y_pos = np.arange(len(variants))
bars = ax.barh(y_pos, accuracies, color=colors_list, height=0.55, edgecolor='none')

ax.set_xlabel('Classification Accuracy on PEMS-SF (%)', fontsize=10.5, fontweight='bold', color='#1E293B')
ax.set_title('Ablation Analysis: Contribution of Architectural Modules', fontsize=12, fontweight='bold', pad=12, color='#0F172A')
ax.set_yticks(y_pos)
ax.set_yticklabels(variants, fontsize=9.5, color='#334155')
ax.set_xlim(80, 92)
ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1')
ax.invert_yaxis()

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.25, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
            ha='left', va='center', fontsize=9, fontweight='bold', color='#1E293B')

plt.tight_layout()
fig.savefig('d:/first intern(maybe)/fig_ablation_breakdown.png', dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# Figure 3: Conceptual System Architecture Flow
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2), dpi=300)
ax.axis('off')

# Draw boxes for architecture flow
boxes = [
    (0.02, 0.35, 0.14, 0.45, 'Multivariate\nTime Series\nX in R^(C x T)', '#E2E8F0', '#475569'),
    (0.20, 0.55, 0.18, 0.38, 'Temporal Stream:\nResTCN (d=1..16)\n+ SE Attention', '#DBEAFE', '#1D4ED8'),
    (0.20, 0.12, 0.18, 0.38, 'Spatial Stream:\nMulti-Scale GCN\n(radii r=1, 3, 5)', '#CCFBF1', '#0F766E'),
    (0.44, 0.55, 0.18, 0.38, 'Hierarchical GoG 1\n(Inner GCN r=3\nOuter GCN r=1)', '#EDE9FE', '#6D28D9'),
    (0.44, 0.12, 0.18, 0.38, 'Hierarchical GoG 2\n(Inner GCN r=3\nOuter GCN r=1)', '#EDE9FE', '#6D28D9'),
    (0.68, 0.25, 0.15, 0.60, '4-Pathway\nSoftmax Fusion\n[w1, w2, w3, w4]', '#FEF3C7', '#B45309'),
    (0.87, 0.35, 0.11, 0.45, 'Classification\nHead\n(2-layer MLP)', '#FEE2E2', '#B91C1C')
]

import matplotlib.patches as mpatches

for x, y, w, h, text, bg, border in boxes:
    box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=bg, edgecolor=border, linewidth=1.5,
                                transform=ax.transAxes, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=border, transform=ax.transAxes, zorder=3)

# Connectors
arrow_kw = dict(arrowstyle="->", lw=1.5, color="#64748B", shrinkA=2, shrinkB=2)
ax.annotate("", xy=(0.20, 0.74), xytext=(0.16, 0.65), xycoords='axes fraction', arrowprops=arrow_kw)
ax.annotate("", xy=(0.20, 0.31), xytext=(0.16, 0.45), xycoords='axes fraction', arrowprops=arrow_kw)
ax.annotate("", xy=(0.44, 0.74), xytext=(0.38, 0.74), xycoords='axes fraction', arrowprops=arrow_kw)
ax.annotate("", xy=(0.44, 0.31), xytext=(0.38, 0.31), xycoords='axes fraction', arrowprops=arrow_kw)

# To fusion: GoG pathways
ax.annotate("", xy=(0.68, 0.62), xytext=(0.62, 0.74), xycoords='axes fraction', arrowprops=arrow_kw)
ax.annotate("", xy=(0.68, 0.38), xytext=(0.62, 0.31), xycoords='axes fraction', arrowprops=arrow_kw)

# Global pool bypass pathways (curved above and below GoG boxes)
ax.annotate("", xy=(0.68, 0.76), xytext=(0.38, 0.85), xycoords='axes fraction',
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748B", ls='--', connectionstyle="arc3,rad=-0.15"))
ax.annotate("", xy=(0.68, 0.24), xytext=(0.38, 0.15), xycoords='axes fraction',
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748B", ls='--', connectionstyle="arc3,rad=0.15"))

# To classification
ax.annotate("", xy=(0.87, 0.55), xytext=(0.83, 0.55), xycoords='axes fraction', arrowprops=arrow_kw)

ax.set_title('ST-GoG End-to-End System Architecture Pipeline', fontsize=12, fontweight='bold', pad=10, color='#0F172A')

plt.tight_layout()
fig.savefig('d:/first intern(maybe)/fig_architecture.png', dpi=300)
plt.close(fig)

print("All figures successfully created!")
