"""
Generate System Architecture Diagram — ZERO OVERLAP, PERFECT TOP HEADROOM
Outputs to: ~/Desktop/vlm/visualization/00_system_architecture_diagram.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

VIS_DIR = os.path.expanduser("~/Desktop/vlm/visualization")
os.makedirs(VIS_DIR, exist_ok=True)

# ─── Dark Mode Color Palette ──────────────────────────────────────────────────
BG          = '#0B0719'
NODE_BG     = '#160E33'
BOX_BORDER  = '#7C4DFF'
ACCENT_CYAN = '#69F0AE'
ACCENT_PINK = '#FF80AB'
ACCENT_GOLD = '#FFD54F'
ACCENT_PURP = '#B388FF'
TEXT_WHITE  = '#F8F0FF'
MUTED_TEXT  = '#BDB8CC'

fig, ax = plt.subplots(figsize=(16, 10.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(-8, 106)  # Headroom on top (-8 to +106) eliminates ALL text overlaps!

# Main Title & Subtitle placed at top with clear separation
ax.text(50, 102.5, 'ASTM E155 NDT Vision LLM & FAISS RAG End-to-End System Architecture',
        ha='center', va='center', fontsize=17, fontweight='bold', color=TEXT_WHITE)
ax.text(50, 98.2, 'Multi-Hop & Cross-Module Communication Stream (Client <-> Bridge <-> AI Core <-> DGX Spark HPC)',
        ha='center', va='center', fontsize=11.2, color=ACCENT_PURP)

# Helper function to draw futuristic node boxes
def draw_node(x, y, w, h, title, subtitle, items, border_color=BOX_BORDER, fill_color=NODE_BG):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                          linewidth=2.0, edgecolor=border_color, facecolor=fill_color)
    ax.add_patch(rect)
    
    # Title Banner
    ax.text(x + w/2, y + h - 4.0, title, ha='center', va='center',
            fontsize=12.2, fontweight='bold', color=border_color)
    if subtitle:
        ax.text(x + w/2, y + h - 7.8, subtitle, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=ACCENT_GOLD)
        
    # Divider line
    ax.plot([x + 2, x + w - 2], [y + h - 10.5, y + h - 10.5], color=border_color, linewidth=0.8, alpha=0.5)
    
    # Bullet items
    start_y = y + h - 15.0
    for i, item in enumerate(items):
        ax.text(x + 2.5, start_y - i * 6.2, f"• {item}", ha='left', va='center',
                fontsize=9.2, color=TEXT_WHITE)

# ─── 4 UNIFORM GRID BOXES (y = 16, h = 63, w = 21.5) ───────────────────────────
UNIFORM_Y = 16
UNIFORM_H = 63
UNIFORM_W = 21.5

# 1. User & Client Layer (MacBook Pro)
draw_node(2.5, UNIFORM_Y, UNIFORM_W, UNIFORM_H,
          "User & Client Layer", "MacBook Pro Workstation",
          ["Interactive Web Inspector", "Streamlit Glassmorphism UI", "Real-Time X-Ray Frame Selector", "Live Verification Certificates", "Bidirectional Chat Interface", "Multi-Frame Comparison"],
          border_color=ACCENT_CYAN)

# 2. Bidirectional Streaming Bridge
draw_node(26.5, UNIFORM_Y, UNIFORM_W, UNIFORM_H,
          "Streaming Bridge", "Bidirectional Pipeline",
          ["WebSocket Connection", "Async Data Serialization", "Live Telemetry Push Stream", "Sub-3.5ms Frame Streaming", "Latency Control Buffer", "Real-Time Event Dispatcher"],
          border_color=ACCENT_PINK)

# 3. Intelligence Engine (Vision LLM & RAG)
draw_node(50.5, UNIFORM_Y, UNIFORM_W, UNIFORM_H,
          "AI Intelligence Core", "Vision LLM & FAISS RAG",
          ["512D Vector Embedding Engine", "Multi-Head Amplified Encoder", "FAISS Vector Index Database", "ASTM E155 Standard Base", "5-Class Defect Classifier", "Ensemble Decision Aggregator"],
          border_color=ACCENT_PURP)

# 4. HPC Compute Layer (NVIDIA DGX Spark)
draw_node(74.5, UNIFORM_Y, UNIFORM_W, UNIFORM_H,
          "HPC Hardware Layer", "NVIDIA DGX Spark",
          ["Grace Blackwell GB10 GPU", "128GB Unified Memory", "332+ FPS Real-Time CUDA", "PyTorch Forward Inference", "Focal Loss & SupCon Model", "Zero-Latency Tensor Core"],
          border_color=ACCENT_GOLD)

# ─── 1. ADJACENT (1-HOP) CONNECTORS (y = 47.5) ────────────────────────────────
def draw_bidirectional_arrows(x1, x2, y_level, top_label, bottom_label, color1=ACCENT_CYAN, color2=ACCENT_PINK):
    arrow_fwd = patches.FancyArrowPatch((x1, y_level + 2.5), (x2, y_level + 2.5),
                                         arrowstyle='->,head_width=3.5,head_length=4.5',
                                         color=color1, linewidth=1.8, mutation_scale=1.2)
    ax.add_patch(arrow_fwd)
    ax.text((x1 + x2)/2, y_level + 4.8, top_label, ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=color1,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=color1, lw=0.7))

    arrow_bwd = patches.FancyArrowPatch((x2, y_level - 2.5), (x1, y_level - 2.5),
                                         arrowstyle='->,head_width=3.5,head_length=4.5',
                                         color=color2, linewidth=1.8, mutation_scale=1.2)
    ax.add_patch(arrow_bwd)
    ax.text((x1 + x2)/2, y_level - 4.8, bottom_label, ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=color2,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=color2, lw=0.7))

draw_bidirectional_arrows(24.0, 26.5, 47.5, "User Request", "Live UI Sync", ACCENT_CYAN, ACCENT_PINK)
draw_bidirectional_arrows(48.0, 50.5, 47.5, "Stream Data", "RAG Results", ACCENT_PINK, ACCENT_PURP)
draw_bidirectional_arrows(72.0, 74.5, 47.5, "CUDA Exec", "Vector Out", ACCENT_PURP, ACCENT_GOLD)

# ─── 2. MULTI-HOP (2-BOX & 3-BOX) CROSS-MODULE ARC ARROWS ─────────────────────
# A) 2-Box Arc: Box 1 (User) <----> Box 3 (AI Core) [Top Arc at y = 79~87, well below subtitle y = 98.2]
arc_2hop = patches.FancyArrowPatch((13.25, 79.0), (61.25, 79.0),
                                   connectionstyle="arc3,rad=-0.26",
                                   arrowstyle='<->,head_width=4.0,head_length=5.0',
                                   color=ACCENT_CYAN, linewidth=2.2)
ax.add_patch(arc_2hop)
ax.text(37.25, 87.5, "Direct RAG Prompt Queries & Vector Retrieval (Box 1 <-> Box 3)",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color=ACCENT_CYAN,
        bbox=dict(boxstyle="round,pad=0.3", fc='#10231C', ec=ACCENT_CYAN, lw=1.0))

# B) 3-Box Arc: Box 1 (User) <----> Box 4 (DGX Spark GPU) [Bottom Arc safely above y = -6]
arc_3hop = patches.FancyArrowPatch((13.25, 16.0), (85.25, 16.0),
                                   connectionstyle="arc3,rad=0.22",
                                   arrowstyle='<->,head_width=4.0,head_length=5.0',
                                   color=ACCENT_GOLD, linewidth=2.2)
ax.add_patch(arc_3hop)
ax.text(49.25, 5.5, "Direct DGX Spark Hardware Telemetry & Real-Time GPU Profiling (Box 1 <-> Box 4)",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color=ACCENT_GOLD,
        bbox=dict(boxstyle="round,pad=0.3", fc='#26200B', ec=ACCENT_GOLD, lw=1.0))

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "00_system_architecture_diagram.png"), dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("🎉 SUCCESS: Regenerated 00_system_architecture_diagram.png ZERO OVERLAP!")
