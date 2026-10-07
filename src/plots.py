"""One chart style for every figure in evals/ (matplotlib, light surface).

Colors come from a validated palette: one blue accent for the model the app uses (or
the first series), orange for a second series, neutral gray for everything else, and a
single-hue blue ramp for heatmaps. Text is always ink-colored, never series-colored.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis labels, ticks
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"       # series 1 / the deployed model
ACCENT_2 = "#eb6834"     # series 2
OTHER = "#b9b7af"        # every other model
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cell_text_color(value: float, vmax: float = 1.0) -> str:
    """Readable annotation color on the blue ramp."""
    return "white" if value / vmax > 0.55 else INK


def heatmap(ax, values, row_labels, col_labels, fmt="{:.2f}", vmax=1.0, title=None):
    """Annotated heatmap on the sequential blue ramp. Returns the image (for a colorbar)."""
    im = ax.imshow(values, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = values[i][j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                    color=cell_text_color(v, vmax))
    if title:
        ax.set_title(title)
    return im
