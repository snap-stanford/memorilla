"""Tables of a profiling run: ``python -m profiling.summarize <out>`` writes ``summary.md`` and ``summary.csv``.

Each row is one cell: the median read time per call with its interquartile range in milliseconds, the median time
per question (read time / batch size), the bandwidth a read achieves if it reads the stored collection or cache once
and its share of the GPU's peak, the time to write the cache (``kv`` settings), memory in MiB (the stored collection
or cache, and PyTorch's peak allocation from compilation or capture through the reads, above the allocation before them) and the output check:
the relative difference from the default forward pass and whether it is within tolerance. The summary lists every
checked cell outside tolerance, and cells that were not checked (batch x N above ``--check_max``) show no verdict.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from profiling.run import RESULTS_FILE

AXES = ("cache", "mode", "shared", "splits", "batch", "n")
MIB = 2**20


def load(out: Path) -> list[dict[str, Any]]:
    """Read the results log; a cell written twice keeps its last record."""
    path = out / RESULTS_FILE
    if not path.exists():
        raise SystemExit(f"No {RESULTS_FILE} in {out}.")
    cells = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            cells[json.dumps(record["key"], sort_keys=True)] = record
    return list(cells.values())


def row(record: dict[str, Any]) -> dict[str, Any]:
    """One flat row of a cell."""
    values: dict[str, Any] = {axis: record["key"][axis] for axis in AXES}
    values["status"] = record["status"]
    read = record.get("timings", {}).get("read")
    if read:
        values.update(read_median_ms=read["median_ms"], read_p25_ms=read["p25_ms"], read_p75_ms=read["p75_ms"])
    values["per_query_ms"] = record.get("per_query_ms")
    values["achieved_gbs"] = record.get("achieved_gbs")
    peak = record.get("peak_bandwidth_gbs")
    values["peak_share"] = values["achieved_gbs"] / peak if peak and values["achieved_gbs"] else None
    values["write_median_ms"] = record["write"]["median_ms"] if "write" in record else None
    values["state_mib"] = record["state_bytes"] / MIB if "state_bytes" in record else None
    peak_memory = record.get("gpu_memory", {}).get("torch_peak_delta")
    values["peak_memory_mib"] = None if peak_memory is None else peak_memory / MIB
    values["rel_diff_default"] = record.get("rel_diff_default")
    values["tolerance"] = record.get("tolerance")
    values["within_tolerance"] = record.get("within_tolerance")
    values["rel_error"] = record.get("rel_error")
    values["default_rel_error"] = record.get("default_rel_error")
    values["reason"] = record.get("reason")
    return values


def fmt(value: Any) -> str:
    """Format a number for a Markdown table."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2e}" if 0 < abs(value) < 1e-3 else f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}"
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)


def markdown(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    """The Markdown summary."""
    first = records[0]
    env = first.get("env", {})
    settings = {name: value for name, value in first["key"].items() if name not in AXES}
    lines = ["# Memory module profile", ""]
    lines += [f"- {name}: {value}" for name, value in settings.items()]
    lines += [f"- {env.get('gpu')}, torch {env.get('torch')}, CUDA {env.get('cuda')}, commit {env.get('git_commit')}"]
    lines += ["", "Read time per call in ms, median [p25, p75]. GB/s assumes the stored collection or cache is read"]
    lines += [
        "once per call; for host it is the host-to-GPU copy rate, so its share of the HBM peak is not meaningful."
    ]
    lines += [
        "The output check compares each cell's output with the default forward pass (see profiling/README.md).",
        "",
    ]
    outside = [r for r in rows if r.get("within_tolerance") is False]
    if outside:
        lines += [
            f"**{len(outside)} checked cells differ from the default forward pass by more than the tolerance:**",
            "",
        ]
        lines += [
            f"- {' '.join(f'{axis}={r[axis]}' for axis in AXES)}: {fmt(r['rel_diff_default'])} > {fmt(r['tolerance'])}"
            for r in outside
        ]
        lines.append("")
    else:
        lines += ["Every checked cell is within tolerance of the default forward pass.", ""]
    header = [
        "cache",
        "mode",
        "shared",
        "splits",
        "batch",
        "N",
        "status",
        "read (ms)",
        "per question (ms)",
        "GB/s",
        "of peak",
        "write (ms)",
        "stored (MiB)",
        "peak (MiB)",
        "diff vs default",
        "tolerance",
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        read = r.get("read_median_ms")
        cells = [fmt(r[axis]) for axis in AXES] + [
            r["status"] if r["status"] == "ok" else f"{r['status']}: {r['reason']}"
        ]
        cells.append("" if read is None else f"{fmt(read)} [{fmt(r['read_p25_ms'])}, {fmt(r['read_p75_ms'])}]")
        cells += [fmt(r.get(name)) for name in ("per_query_ms", "achieved_gbs")]
        cells.append("" if r.get("peak_share") is None else f"{r['peak_share']:.0%}")
        cells += [fmt(r.get(name)) for name in ("write_median_ms", "state_mib", "peak_memory_mib", "rel_diff_default")]
        verdict = r.get("within_tolerance")
        cells.append("" if verdict is None else f"{'ok' if verdict else 'OUTSIDE'} ({fmt(r['tolerance'])})")
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def main(argv: list[str] | None = None) -> None:
    """Write the summary of one output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="Output directory of profiling.run.")
    args = parser.parse_args(argv)

    records = load(args.out)
    rows = [row(record) for record in records]
    (args.out / "summary.md").write_text("\n".join(markdown(records, rows)))
    fields = list(dict.fromkeys(name for r in rows for name in r))
    with open(args.out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    outside = sum(r.get("within_tolerance") is False for r in rows)
    checked = sum(r.get("within_tolerance") is not None for r in rows)
    print(f"wrote {args.out / 'summary.md'} and {args.out / 'summary.csv'}")
    print(f"output check: {checked - outside} of {checked} checked cells within tolerance of the default forward pass")


if __name__ == "__main__":
    main()
