"""Plot how the memory module scales with N: ``python -m profiling.plot <out> <figure.png>``.

Reads ``<out>/memory_module.jsonl`` and draws two panels for one question per call: (a) the time per question of the
default forward pass, the memory cache and the cache with ``torch.compile``, and (b) the GPU memory held by the stored
embeddings and by the cache. The figure format follows the file extension (PNG, PDF or SVG). Needs matplotlib
(``pip install matplotlib``).
"""

import argparse
import json
from pathlib import Path
from typing import Any

from profiling.run import RESULTS_FILE

# Categorical slots 1-3 of the dataviz reference palette, validated for color-vision deficiencies.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LATENCY_SERIES = (
    ("Default forward pass", "gpu", "eager", BLUE, "o"),
    ("Memory cache", "kv", "eager", ORANGE, "s"),
    ("Memory cache + torch.compile", "kv", "compile", AQUA, "^"),
)
MEMORY_SERIES = (("Document embeddings", "gpu", BLUE, "o"), ("Memory cache", "kv", ORANGE, "s"))
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
GIB = 2**30
ANNOTATE_N = 2**20  # the speedup is marked at about one million documents
COUNT_TICKS = {1: "1", 10: "10", 100: "100", 10**3: "1K", 10**4: "10K", 10**5: "100K", 10**6: "1M", 10**7: "10M"}
MS_TICKS = {0.1: "0.1", 1: "1", 10: "10", 100: "100", 1000: "1,000"}
BYTE_TICKS = {10 * 2**10: "10 KiB", 2**20: "1 MiB", 100 * 2**20: "100 MiB", 10 * GIB: "10 GiB"}
STYLE = {
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
}


def load(out: Path) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    """Successful one-question cells by (cache, mode), then by N."""
    cells: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for line in (out / RESULTS_FILE).read_text().splitlines():
        record = json.loads(line)
        key = record["key"]
        if record["status"] == "ok" and key["batch"] == 1 and not key["shared"]:
            cells.setdefault((key["cache"], key["mode"]), {})[key["n"]] = record
    return cells


def total_memory(record: dict[str, Any]) -> int | None:
    """Total memory of the GPU a record came from, in bytes, from its ``nvidia-smi`` line."""
    fields = (record["env"].get("nvidia_smi") or "").split(",")
    return int(fields[1].split()[0]) * 2**20 if len(fields) > 1 and fields[1].strip().endswith("MiB") else None


def style(axis: Any, letter: str, title: str, ylabel: str, yticks: dict[float, str]) -> None:
    """Log-log axes with readable ticks, a recessive grid and a panel letter."""
    axis.set_xscale("log")
    axis.set_yscale("log")
    low, high = axis.get_ylim()
    shown = {value: text for value, text in yticks.items() if low <= value <= high}
    axis.set_xticks(list(COUNT_TICKS), list(COUNT_TICKS.values()))
    axis.set_yticks(list(shown), list(shown.values()))
    axis.minorticks_off()
    axis.set_xlim(0.5, 4e7)  # room for the end labels
    axis.set_xlabel("Documents in the collection (N)")
    axis.set_ylabel(ylabel)
    axis.set_title(f"$\\bf{{{letter}}}$   {title}", color=TEXT, loc="left", pad=10)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_line(axis: Any, points: list[tuple[int, float]], color: str, marker: str, label: str, value: str) -> None:
    """A 2px line with white-edged markers and its last value labelled at the end."""
    ns, values = zip(*points, strict=True)
    axis.plot(
        ns, values, color=color, linewidth=2, marker=marker, markersize=6.5, markeredgecolor=SURFACE,
        markeredgewidth=1, label=label, solid_capstyle="round", zorder=3,
    )  # fmt: skip
    axis.annotate(value, (ns[-1], values[-1]), xytext=(7, 0), textcoords="offset points", va="center", color=TEXT)


def ratio_bracket(axis: Any, x: float, low: float, high: float, text: str, height: float = 0.5) -> None:
    """A vertical bracket from ``low`` to ``high`` at ``x``, labelled on its left.

    ``height`` places the label between the ends on a log scale (0 at ``low``, 1 at ``high``).
    """
    axis.annotate(
        "", (x, low), (x, high), arrowprops={"arrowstyle": "<->", "color": MUTED, "linewidth": 1, "shrinkA": 0,
                                             "shrinkB": 0},
    )  # fmt: skip
    axis.annotate(text, (x, low ** (1 - height) * high**height), xytext=(-6, 0), textcoords="offset points",
                  ha="right", va="center", color=TEXT, fontweight="bold", zorder=4,
                  bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    """Draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="Output directory of profiling.run.")
    parser.add_argument("figure", type=Path, help="Figure to write, e.g. assets/efficiency.png.")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    cells = load(args.out)
    first = next(iter(cells[("gpu", "eager")].values()))
    figure, (latency, memory) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)

    times = {}
    for label, cache, mode, color, marker in LATENCY_SERIES:
        points = [(n, record["timings"]["read"]["median_ms"]) for n, record in sorted(cells[(cache, mode)].items())]
        times[(cache, mode)] = dict(points)
        plot_line(latency, points, color, marker, label, f"{points[-1][1]:.3g} ms")
    default, fastest = times[("gpu", "eager")][ANNOTATE_N], times[("kv", "compile")][ANNOTATE_N]
    ratio_bracket(latency, ANNOTATE_N, fastest, default, f"{default / fastest:.1f}×", height=0.32)
    style(latency, "a", "Latency per question", "Time (ms)", MS_TICKS)
    latency.legend(frameon=False, labelcolor=TEXT, loc="upper left")

    sizes = {
        cache: sorted((n, r["state_bytes"]) for n, r in cells[(cache, "eager")].items()) for cache in ("gpu", "kv")
    }
    ratio = sizes["kv"][-1][1] / sizes["gpu"][-1][1]
    for label, cache, color, marker in MEMORY_SERIES:
        points = sizes[cache]
        value = f"{points[-1][1] / GIB:.0f} GiB" + (f" ({ratio:.0f}×)" if cache == "kv" else "")
        plot_line(memory, points, color, marker, label, value)
    capacity = total_memory(first)
    memory.set_yscale("log")
    if capacity is not None:
        memory.axhline(capacity, color=MUTED, linewidth=1, zorder=2)
        memory.annotate(
            f"{first['env']['gpu']} memory ({capacity / GIB:.0f} GiB)", (1, capacity), xytext=(0, 5),
            textcoords="offset points", color=MUTED,
        )  # fmt: skip
        lowest = min(min(line.get_ydata()) for line in memory.get_lines())
        memory.set_ylim(lowest / 3, capacity * 6)
    style(memory, "b", "GPU memory of the collection", "Memory", BYTE_TICKS)
    memory.legend(frameon=False, labelcolor=TEXT, loc="lower right")

    figure.tight_layout(w_pad=3)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
