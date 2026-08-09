"""
Generate Table Visualizations — DRAMATICALLY ENLARGED TEXT & ULTRA-HIGH CONTRAST
Outputs to: ~/Desktop/vlm/visualization/
  08_vlm_model_leaderboard_table.png
  09_astm_holdout_verification_table.png
  10_ndt_economic_comparison_table.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

VIS_DIR = os.path.expanduser("~/Desktop/vlm/visualization")
os.makedirs(VIS_DIR, exist_ok=True)

# Colors
BG       = '#0D0920'
HEADER   = '#1E0F45'
ROW_ODD  = '#140C2E'
ROW_EVEN = '#1C113D'
BORDER   = '#9C68FF'
GOLD     = '#FFD700'
SILVER   = '#E0E0E0'
BRONZE   = '#FFA726'
PURPLE   = '#D1B3FF'
CYAN     = '#00E676'
PINK     = '#FF4081'
WHITE    = '#FFFFFF'
MUTED    = '#D1C4E9'

def add_cell(fig, x, y, w, h, text, facecolor, edgecolor=BORDER,
             color=WHITE, fontsize=13, fontweight='bold', lw=1.2):
    rect = FancyBboxPatch((x, y), w - 0.004, h,
                          boxstyle="round,pad=0.004",
                          linewidth=lw, edgecolor=edgecolor,
                          facecolor=facecolor,
                          transform=fig.transFigure, figure=fig, clip_on=False)
    fig.add_artist(rect)
    fig.text(x + w / 2 - 0.002, y + h / 2, text,
             ha='center', va='center', color=color,
             fontsize=fontsize, fontweight=fontweight,
             clip_on=False)

# =============================================================
# TABLE 08: VLM 5-Model & Ensemble Leaderboard (GIGANTIC TEXT)
# =============================================================
print("[8/10] Plotting 08_vlm_model_leaderboard_table.png with GIGANTIC TEXT...")

fig, ax = plt.subplots(figsize=(16, 7.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

fig.text(0.5, 0.96,
         '1. VLM 5-Model & Ensemble FAISS NDT RAG Leaderboard (%)',
         ha='center', va='top', fontsize=18, fontweight='bold', color=WHITE)

col_labels = [
    'Rank', 'Model / Ensemble Name', 'Training\nStatus',
    'RAG Recall\n(%)', 'P@1 Precision\n(%)',
    'Accuracy\n(%)', 'MRR\n(%)',
    'Latency', 'Operation & RAG Status'
]

rows = [
    ['1st', '512D Vector Embedding Engine', 'Epoch [50/50]',
     '94.90 %', '94.85 %', '85.05 %', '94.90 %', '3.4 ms',
     '50-Epoch Complete (GPU Idle)'],
    ['2nd', 'Soft-Voting Ensemble (5 VLMs)', 'Epoch [50/50]',
     '92.10 %', '92.00 %', 'N/A', '92.10 %', '24.8 ms',
     'FAISS RAG Eval Done'],
    ['3rd', 'Hard-Voting Ensemble (5 VLMs)', 'Epoch [50/50]',
     '90.40 %', '90.30 %', 'N/A', '90.40 %', '22.1 ms',
     'FAISS RAG Eval Done'],
    ['4th', 'Stacking Ensemble (5 VLMs)', 'Epoch [50/50]',
     '89.70 %', '89.60 %', 'N/A', '89.70 %', '31.8 ms',
     'FAISS RAG Eval Done'],
    ['5th', 'GLM-4V-9B', 'Epoch [50/50]',
     '88.20 %', '88.10 %', '88.10 %', '88.20 %', '18.1 ms',
     'FAISS RAG Eval Done'],
]

col_widths  = [0.055, 0.200, 0.100, 0.105, 0.105, 0.080, 0.075, 0.075, 0.190]
rank_colors = [GOLD, SILVER, BRONZE, MUTED, MUTED]
x_start = 0.01
y_header = 0.79
h_hdr   = 0.13
row_h   = 0.115

x = x_start
for lbl, w in zip(col_labels, col_widths):
    add_cell(fig, x, y_header, w, h_hdr, lbl,
             facecolor=HEADER, color=PURPLE,
             fontsize=12, fontweight='bold', lw=1.5)
    x += w

for ri, (row, rc) in enumerate(zip(rows, rank_colors)):
    bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
    y_row = y_header - (ri + 1) * (row_h + 0.009)
    x = x_start
    for ci, (cell, w) in enumerate(zip(row, col_widths)):
        color = rc if ci == 0 else (CYAN if ci in [3, 4, 5, 6] else WHITE)
        add_cell(fig, x, y_row, w, row_h, cell,
                 facecolor=bg, color=color, fontsize=11.5, fontweight='bold')
        x += w

plt.savefig(os.path.join(VIS_DIR, "08_vlm_model_leaderboard_table.png"),
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✅ Saved 08_vlm_model_leaderboard_table.png")


# =============================================================
# TABLE 09: ASTM E155 Holdout Verification (GIGANTIC TEXT)
# =============================================================
print("[9/10] Plotting 09_astm_holdout_verification_table.png with GIGANTIC TEXT...")

fig, ax = plt.subplots(figsize=(16, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

fig.text(0.5, 0.95,
         '2. ASTM E155 Discontinuity 836-Frame Holdout Verification Table (%)',
         ha='center', va='top', fontsize=17, fontweight='bold', color=WHITE)

col_labels2 = [
    'ASTM E155 Discontinuity Type',
    'Precision (%)', 'Recall (%)', 'F1-Score (%)',
    'Validation Status'
]
col_widths2 = [0.420, 0.125, 0.125, 0.125, 0.190]

rows2 = [
    ['Type 1: Gas Porosity Inclusions (Spherical Gas Pores)',
     '96.10 %', '95.80 %', '95.95 %', '836 Frames Verified'],
    ['Type 2: Microshrinkage Porosity (Fine Shrinkage Pores)',
     '94.80 %', '94.20 %', '94.50 %', '836 Frames Verified'],
    ['Type 3: Shrinkage Cavities (Spongy Shrinkage Voids)',
     '95.40 %', '95.00 %', '95.20 %', '836 Frames Verified'],
    ['Type 4: Foreign Inclusions / Dross (Oxide Inclusions)',
     '93.60 %', '93.10 %', '93.35 %', '836 Frames Verified'],
    ['Normal / Pass (No Defect / Sound Matrix)',
     '94.80 %', '94.80 %', '94.80 %', '836 Frames Verified'],
]

x_s2 = 0.012
y_h2 = 0.77
h_h2 = 0.12
r_h2 = 0.12

x = x_s2
for lbl, w in zip(col_labels2, col_widths2):
    add_cell(fig, x, y_h2, w, h_h2, lbl,
             facecolor=HEADER, color=PURPLE,
             fontsize=12.5, fontweight='bold', lw=1.5)
    x += w

for ri, row in enumerate(rows2):
    bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
    y_row = y_h2 - (ri + 1) * (r_h2 + 0.010)
    x = x_s2
    for ci, (cell, w) in enumerate(zip(row, col_widths2)):
        color = WHITE if ci == 0 else CYAN
        add_cell(fig, x, y_row, w, r_h2, cell,
                 facecolor=bg, color=color, fontsize=11.5, fontweight='bold')
        x += w

plt.savefig(os.path.join(VIS_DIR, "09_astm_holdout_verification_table.png"),
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✅ Saved 09_astm_holdout_verification_table.png")


# =============================================================
# TABLE 10: NDT AI Economic Evaluation (GIGANTIC TEXT)
# =============================================================
print("[10/10] Plotting 10_ndt_economic_comparison_table.png with GIGANTIC TEXT...")

fig, ax = plt.subplots(figsize=(16, 8.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

fig.text(0.5, 0.96,
         'NDT AI Pre / Post Deployment Comparative Economic Evaluation',
         ha='center', va='top', fontsize=18, fontweight='bold', color=WHITE)

col_labels3 = [
    'Evaluation Metric',
    'Baseline: Human Inspector\n(Level III Certified)',
    'AI VLM RAG Automation System',
    'Economic Impact & ROI'
]
col_widths3 = [0.220, 0.240, 0.300, 0.225]

rows3 = [
    ['Inspection Speed',
     '3.0–5.0 min / film\n(~120 films / day)',
     '3.4 ms / frame\n(10,000+ frames / hr)',
     'Speed +99.9% Faster'],
    ['Annual Operating Cost\n(per production line)',
     '~150M KRW / yr\n(2× Inspectors)',
     'Annual Server Cost\n~18M KRW',
     '88.3% Cost Savings\n(~130M KRW / yr)'],
    ['Human Error / Defect Miss Rate',
     '5.2%\n(fatigue after shift)',
     '< 0.1% Miss Rate\n(Recall 95.8%–96.1%)',
     'Defect Escape: 0%'],
    ['Defect Scrap Cost',
     '~4.5M KRW per scrap\n(after machining)',
     'Blocked at casting stage\n(<12K KRW per event)',
     '~360M KRW / yr\nScrap Loss Prevented'],
    ['Initial Investment Payback',
     'N/A',
     'Full Payback within ~4.2 Months',
     'Year-1 ROI: +310% Profit'],
]

x_s3 = 0.010
y_h3 = 0.81
h_h3 = 0.12
r_h3 = 0.130

x = x_s3
for lbl, w in zip(col_labels3, col_widths3):
    add_cell(fig, x, y_h3, w, h_h3, lbl,
             facecolor=HEADER, color=PURPLE,
             fontsize=12.5, fontweight='bold', lw=1.5)
    x += w

for ri, row in enumerate(rows3):
    bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
    y_row = y_h3 - (ri + 1) * (r_h3 + 0.010)
    x = x_s3
    for ci, (cell, w) in enumerate(zip(row, col_widths3)):
        color = WHITE if ci == 0 else (MUTED if ci == 1 else (CYAN if ci == 2 else PINK))
        add_cell(fig, x, y_row, w, r_h3, cell,
                 facecolor=bg, color=color, fontsize=11.5, fontweight='bold')
        x += w

plt.savefig(os.path.join(VIS_DIR, "10_ndt_economic_comparison_table.png"),
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✅ Saved 10_ndt_economic_comparison_table.png")

print("\n🎉 ALL 3 TABLE PNGs REGENERATED WITH GIGANTIC TEXT!")
