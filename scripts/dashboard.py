"""Generate a self-contained HTML progress page: project steps and live training curves.

Usage:
    uv run python scripts/dashboard.py            # write reports/progress.html once
    uv run python scripts/dashboard.py --watch    # rewrite every 30 s (page auto-refreshes)
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

from chessbot.config import REPO_ROOT, TrainConfig, load_config

OUT = REPO_ROOT / "reports" / "progress.html"

# (label, what it produced). Status for Step 5 is derived from the training log.
STEPS = [
    ("Step 0", "Success & measurement", "Evaluation protocol written (docs/evaluation-protocol.md)"),
    ("Step 1", "Environment & tooling", "uv project, configs, Stockfish 19, MPS GPU verified"),
    ("Step 2", "State & move encoding", "19-plane board, 8x8x73 moves; round-trip tested on 10k+ positions"),
    ("Step 3", "Baselines", "Random, greedy, alpha-beta bots + Stockfish match runner + Elo CIs"),
    ("Step 4", "Network", "ResNet 6x64, 505k params, ~27k positions/s on the Mac GPU"),
    ("Step 5", "Supervised pretraining", "150k Lichess 2000+ games, 10.75M positions"),
    ("Step 6", "MCTS search", "Not started"),
    ("Step 7", "Self-play RL loop", "Not started"),
    ("Step 8", "Rigorous evaluation", "Not started"),
    ("Step 9", "Lichess deployment", "Not started (needs a fresh Lichess account)"),
    ("Step 10", "Resume packaging", "Not started"),
]


def read_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chart(
    title: str, series: list[tuple[str, str, list[tuple[float, float]]]], x_max: float, fmt: str
) -> str:
    """Inline SVG line chart. series = [(label, css colour var, [(x, y), ...])]."""
    w, h, left, bottom, top, right = 560, 220, 48, 28, 12, 12
    ys = [y for _, _, pts in series for _, y in pts]
    if not ys:
        return f"<div class='chart'><h3>{title}</h3><p class='muted'>No data yet.</p></div>"
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.1 or 0.05
    lo, hi = lo - pad, hi + pad

    def px(x: float) -> float:
        return left + (w - left - right) * x / x_max

    def py(y: float) -> float:
        return top + (h - top - bottom) * (1 - (y - lo) / (hi - lo))

    grid = ""
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        grid += (
            f"<line x1='{left}' x2='{w - right}' y1='{py(v):.1f}' y2='{py(v):.1f}' class='grid'/>"
            f"<text x='{left - 6}' y='{py(v) + 4:.1f}' class='ax' text-anchor='end'>{v:{fmt}}</text>"
        )
    for i in range(5):
        x = x_max * i / 4
        grid += f"<text x='{px(x):.1f}' y='{h - 8}' class='ax' text-anchor='middle'>{x / 1000:.0f}k</text>"
    lines, legend = "", ""
    for label, colour, pts in series:
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        lines += f"<polyline points='{d}' fill='none' stroke='var({colour})' stroke-width='2'/>"
        if pts:
            lx, ly = pts[-1]
            lines += f"<circle cx='{px(lx):.1f}' cy='{py(ly):.1f}' r='3.5' fill='var({colour})'/>"
            legend += (
                f"<span class='key'><i style='background:var({colour})'></i>{label} "
                f"<b>{ly:{fmt}}</b></span>"
            )
    return (
        f"<div class='chart'><h3>{title}</h3><div class='legend'>{legend}</div>"
        f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{title}'>{grid}{lines}</svg></div>"
    )


def build() -> str:
    cfg = load_config(TrainConfig, "train")
    ckpt = REPO_ROOT / cfg.checkpoint_dir
    metrics = read_metrics(ckpt / "metrics.jsonl")
    meta_path = REPO_ROOT / cfg.data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    batch = cfg.batch_size
    positions = meta.get("positions", 0)
    train_positions = int(positions * 0.98)
    total = cfg.max_steps or (train_positions // batch) * cfg.epochs
    last = metrics[-1] if metrics else None
    step = int(last["step"]) if last else 0
    pct = 100 * step / total if total else 0
    done = bool(total) and step >= total
    updated = time.strftime("%H:%M:%S")

    def pts(key: str) -> list[tuple[float, float]]:
        return [(m["step"], m[key]) for m in metrics]

    charts = (
        chart(
            "Move prediction accuracy (held-out games)",
            [("top-1", "--c1", pts("top1")), ("top-5", "--c2", pts("top5"))],
            max(total, 1),
            ".1%",
        )
        + chart(
            "Policy loss (lower is better)",
            [("policy", "--c3", pts("policy_loss"))],
            max(total, 1),
            ".2f",
        )
        + chart(
            "Value error (MSE, lower is better)",
            [("value", "--c4", pts("value_mse"))],
            max(total, 1),
            ".2f",
        )
    )

    rows = ""
    for label, name, note in STEPS:
        if label == "Step 5":
            status, cls = ("done", "done") if done else (f"training {pct:.0f}%", "run")
        elif note.startswith("Not started"):
            status, cls = "pending", "todo"
        else:
            status, cls = "done", "done"
        rows += (
            f"<tr><td><span class='pill {cls}'>{status}</span></td>"
            f"<td><b>{label}</b> {html.escape(name)}</td><td class='muted'>{html.escape(note)}</td></tr>"
        )

    stat = lambda v, k: f"<div class='stat'><b>{v}</b><span>{k}</span></div>"  # noqa: E731
    stats = (
        stat(f"{step:,} / {total:,}", "training steps")
        + stat(f"{last['top1']:.1%}" if last else "-", "top-1 move accuracy")
        + stat(f"{last['top5']:.1%}" if last else "-", "top-5 move accuracy")
        + stat(f"{positions:,}", "training positions")
        + stat(f"{meta.get('games', 0):,}", "games (both players 2000+)")
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Chess Bot Progress</title>
<style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;
--c1:#2563eb;--c2:#7c3aed;--c3:#dc2626;--c4:#059669;--ok:#16a34a;--run:#d97706}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;
--card:#1c1917;--line:#292524;--c1:#60a5fa;--c2:#a78bfa;--c3:#f87171;--c4:#34d399;
--ok:#4ade80;--run:#fbbf24}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px;font-size:24px}} h3{{margin:0 0 6px;font-size:14px}}
.muted{{color:var(--muted)}}
.bar{{height:10px;background:var(--line);border-radius:5px;overflow:hidden;margin:12px 0 20px}}
.bar i{{display:block;height:100%;width:{pct:.1f}%;background:var(--run)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}}
.stat,.chart,table{{background:var(--card);border:1px solid var(--line);border-radius:10px}}
.stat{{padding:12px 14px}} .stat b{{display:block;font-size:20px}}
.stat span{{color:var(--muted);font-size:12px}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-bottom:20px}}
.chart{{padding:12px 14px}} svg{{width:100%;height:auto}}
.grid{{stroke:var(--line)}} .ax{{fill:var(--muted);font-size:11px}}
.legend{{display:flex;gap:14px;font-size:12px;margin-bottom:4px}}
.key i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
table{{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden}}
td{{padding:9px 12px;border-top:1px solid var(--line);vertical-align:top}} tr:first-child td{{border-top:0}}
.pill{{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;border:1px solid}}
.done{{color:var(--ok)}} .run{{color:var(--run)}} .todo{{color:var(--muted)}}
</style></head><body><main>
<h1>Chess Bot Progress</h1>
<div class="muted">Updated {updated}, auto-refreshes every 30 s. Training status:
<b>{"finished" if done else "running"}</b> ({pct:.0f}%). Metrics are logged every {cfg.eval_every:,} steps.</div>
<div class="bar"><i></i></div>
<div class="stats">{stats}</div>
<div class="charts">{charts}</div>
<h3>Project steps</h3>
<table>{rows}</table>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    while True:
        OUT.write_text(build())
        print(f"wrote {OUT}")
        if not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
