"""Generate fig_architecture.png for the PAAMS 2026 paper."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe

# ── colours ──────────────────────────────────────────────────────────────────
NAVY      = '#1E3A5F'
WHITE     = '#FFFFFF'
GREY_BOX  = '#E8EAED'
BORDER    = '#3B6FB5'
ARROW_C   = '#7A8FA6'
RED       = '#DC2626'

# ── layout constants ─────────────────────────────────────────────────────────
BOX_W, BOX_H = 5.0, 1.65
BOX_X        = 0.35
GAP          = 0.42          # vertical gap between boxes
JSON_X       = 6.55
JSON_W       = 2.85
N            = 6

fig = plt.figure(figsize=(7.0, 10.5))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0.8, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

# bottom-left y of each box, top -> bottom
# tops[0]+BOX_H = 13.65 (leaves 0.35 top margin within ylim=14)
tops = [12.00 - i * (BOX_H + GAP) for i in range(N)]

# ── helpers ──────────────────────────────────────────────────────────────────
def box(x, y, w, h, fc=NAVY, ec=NAVY, lw=1.2, radius=0.18):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       zorder=3)
    ax.add_patch(p)

def arrow(x0, y0, x1, y1, color=ARROW_C, lw=2.2, ms=14):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms),
                zorder=4)

def txt(x, y, s, size=10, weight='bold', color=WHITE, va='center', ha='center', rot=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color, rotation=rot, zorder=5)

# ── icon drawing functions ────────────────────────────────────────────────────
def icon_waveform(cx, cy, spike=False):
    """ECG-style waveform."""
    xs = np.array([-0.44,-0.28,-0.18,-0.08, 0.00, 0.06, 0.16, 0.26, 0.44])
    ys = np.array([  0.0,  0.1,  0.28,-0.18, 0.0,  0.55,-0.12, 0.0,  0.0])
    sx = cx + xs * 1.05
    sy = cy + ys * 0.55
    if spike:
        ax.plot(sx[:5],  sy[:5],  color=WHITE, lw=2,   solid_capstyle='round', zorder=5)
        ax.plot(sx[4:7], sy[4:7], color=RED,   lw=2.5, solid_capstyle='round', zorder=5)
        ax.plot(sx[6:],  sy[6:],  color=WHITE, lw=2,   solid_capstyle='round', zorder=5)
    else:
        ax.plot(sx, sy, color=WHITE, lw=2, solid_capstyle='round', zorder=5)
    # baseline
    ax.plot([cx-0.46, cx+0.46], [cy-0.30, cy-0.30],
            color=WHITE, lw=1.4, alpha=0.55, zorder=5)

def icon_scatter(cx, cy):
    """Scatter plot with regression line."""
    dx = np.array([-0.30,-0.20,-0.10, 0.00, 0.10, 0.20, 0.28,-0.22, 0.05, 0.24])
    dy = np.array([-0.22,-0.10, 0.02, 0.08, 0.04, 0.18, 0.24,-0.16, 0.05, 0.20])
    ax.scatter(cx + dx, cy + dy * 0.72, s=22, color=WHITE, zorder=6, alpha=0.92)
    ax.plot([cx-0.34, cx+0.34], [cy-0.26, cy+0.26],
            color=WHITE, lw=2.4, zorder=5)
    ax.plot([cx-0.38, cx-0.38], [cy-0.30, cy+0.30],
            color=WHITE, lw=1.5, alpha=0.65, zorder=5)
    ax.plot([cx-0.38, cx+0.38], [cy-0.30, cy-0.30],
            color=WHITE, lw=1.5, alpha=0.65, zorder=5)

def icon_speedometer(cx, cy):
    """Speedometer dial + lightning bolt."""
    # arc
    theta = np.linspace(np.pi * 0.1, np.pi * 0.9, 60)
    r = 0.30
    ox, oy = cx - 0.08, cy - 0.02
    ax.plot(ox + r * np.cos(theta), oy + r * np.sin(theta),
            color=WHITE, lw=2.4, zorder=5)
    # tick marks
    for a in np.linspace(np.pi * 0.15, np.pi * 0.85, 5):
        ax.plot([ox + r*0.82*np.cos(a), ox + r*np.cos(a)],
                [oy + r*0.82*np.sin(a), oy + r*np.sin(a)],
                color=WHITE, lw=1.5, zorder=5)
    # needle
    na = np.pi * 0.62
    ax.annotate('', xy=(ox + r*0.68*np.cos(na), oy + r*0.68*np.sin(na)),
                xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=WHITE, lw=2, mutation_scale=10),
                zorder=6)
    # lightning bolt
    bx = cx + 0.22
    by = cy
    vx = np.array([ 0.00, 0.09,-0.01, 0.11])
    vy = np.array([ 0.28, 0.04, 0.04,-0.22])
    ax.plot(bx + vx * 0.8, by + vy * 0.85,
            color=WHITE, lw=2.4, solid_joinstyle='round',
            solid_capstyle='round', zorder=5)

def icon_database(cx, cy):
    """Stacked cylinder + magnifying glass."""
    ox, ew, eh = cx - 0.14, 0.26, 0.09
    for yo in [0.20, 0.00, -0.20]:
        ell = mpatches.Ellipse((ox, cy + yo), ew*2, eh*2,
                               facecolor=NAVY, edgecolor=WHITE, lw=1.8, zorder=5)
        ax.add_patch(ell)
    ax.fill_between([ox-ew, ox+ew], [cy-0.20]*2, [cy+0.20]*2,
                    color=NAVY, zorder=4)
    ax.plot([ox-ew, ox-ew], [cy-0.20, cy+0.20], color=WHITE, lw=1.8, zorder=5)
    ax.plot([ox+ew, ox+ew], [cy-0.20, cy+0.20], color=WHITE, lw=1.8, zorder=5)
    # top ellipse filled white
    ell_top = mpatches.Ellipse((ox, cy+0.20), ew*2, eh*2,
                               facecolor=WHITE, edgecolor=WHITE, lw=1, zorder=6)
    ax.add_patch(ell_top)
    # magnifying glass
    mgx, mgy, mgr = cx + 0.26, cy + 0.10, 0.13
    circ = plt.Circle((mgx, mgy), mgr, fill=False, color=WHITE, lw=2.4, zorder=6)
    ax.add_patch(circ)
    a = -np.pi/4
    ax.plot([mgx + mgr*np.cos(a), mgx + mgr*1.65*np.cos(a)],
            [mgy + mgr*np.sin(a), mgy + mgr*1.65*np.sin(a)],
            color=WHITE, lw=3.0, solid_capstyle='round', zorder=6)

def icon_document(cx, cy):
    """Document with horizontal lines + quill."""
    dw, dh = 0.24, 0.36
    ox = cx - 0.12
    doc = FancyBboxPatch((ox - dw, cy - dh), dw*2, dh*2,
                         boxstyle="square,pad=0",
                         facecolor=NAVY, edgecolor=WHITE, lw=2, zorder=5)
    ax.add_patch(doc)
    # dog-ear
    cs = 0.09
    corn = plt.Polygon([[ox+dw-cs, cy+dh],
                         [ox+dw,    cy+dh-cs],
                         [ox+dw-cs, cy+dh-cs]],
                        closed=True, facecolor=WHITE, edgecolor=WHITE, zorder=6)
    ax.add_patch(corn)
    # text lines
    for lyo, llen in zip([0.18, 0.05, -0.08, -0.21], [0.20, 0.20, 0.20, 0.13]):
        ax.plot([ox - llen, ox + llen], [cy + lyo]*2,
                color=WHITE, lw=1.7, alpha=0.88, zorder=6)
    # quill
    qx, qy = cx + 0.30, cy
    ax.plot([qx-0.07, qx+0.10], [qy-0.22, qy+0.26],
            color=WHITE, lw=2.0, solid_capstyle='round', zorder=6)
    for t in np.linspace(0.1, 0.9, 5):
        px = qx - 0.07 + t*0.17
        py = qy - 0.22 + t*0.48
        for sgn in [-1, 1]:
            ax.plot([px, px + sgn*0.07], [py, py - 0.10],
                    color=WHITE, lw=1.1, alpha=0.80, zorder=6)

ICONS = [
    lambda cx, cy: icon_waveform(cx, cy, spike=False),  # Data
    lambda cx, cy: icon_scatter(cx, cy),                # Herd-Size
    lambda cx, cy: icon_speedometer(cx, cy),            # Energy
    lambda cx, cy: icon_waveform(cx, cy, spike=True),   # Anomaly
    lambda cx, cy: icon_database(cx, cy),               # RAG
    lambda cx, cy: icon_document(cx, cy),               # LLM
]

LABELS = [
    'Data Agent',
    'Herd-Size Agent',
    'Energy Agent',
    'Anomaly Agent',
    'RAG Agent',
    'LLM Narrative Agent',
]

# ── JSON rectangle ────────────────────────────────────────────────────────────
jy_bot = tops[-1] - 0.12
jy_top = tops[0] + BOX_H + 0.12
box(JSON_X, jy_bot, JSON_W, jy_top - jy_bot,
    fc=GREY_BOX, ec=BORDER, lw=2.5, radius=0.20)
mid_y = jy_bot + (jy_top - jy_bot) * 0.46
txt(JSON_X + JSON_W/2, mid_y,
    'Shared JSON\nReport Object',
    size=11, color=NAVY, rot=90)

# ── agent boxes ───────────────────────────────────────────────────────────────
for i, (y, label, icon_fn) in enumerate(zip(tops, LABELS, ICONS)):
    box(BOX_X, y, BOX_W, BOX_H, radius=0.18)

    # icon on left third of box
    icon_fn(BOX_X + BOX_W * 0.22, y + BOX_H * 0.50)

    # label on right portion, vertically centred
    txt(BOX_X + BOX_W * 0.62, y + BOX_H * 0.50, label, size=11.0)

    # downward arrow to next box
    if i < N - 1:
        ax_mid = BOX_X + BOX_W/2
        arrow(ax_mid, y, ax_mid, tops[i+1] + BOX_H, lw=2.4)

    # horizontal arrow → JSON
    arr_y = y + BOX_H/2
    arrow(BOX_X + BOX_W, arr_y, JSON_X, arr_y, lw=2.4)

# ── save PNG + vector PDF ─────────────────────────────────────────────────────
base = r'C:\Users\22238233\Documents\WP3\demo\paper\fig_architecture'
plt.savefig(base + '.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(base + '.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved PNG and PDF")
