"""Generate a self-contained HTML progress page: project steps and live training curves.

Usage:
    uv run python scripts/dashboard.py            # write reports/progress.html once
    uv run python scripts/dashboard.py --watch    # rewrite every 30 s (page auto-refreshes)
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
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
    ("Step 6", "MCTS search", "Batched PUCT search + position cache; ~1,000 sims/s; mate puzzles pass"),
    ("Step 7", "Self-play RL loop", "Built and tested; first real run pending"),
    ("Step 8", "Rigorous evaluation", "Built and tested: Stockfish harness, rating fit, chart; not yet run"),
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


RESULT_LINE = re.compile(
    r"^(\S+) vs (\S+)\s+\+(\d+) =(\d+) -(\d+) \(([\d.]+)%, Elo diff ([+-]\d+) "
    r"\[([+-]\d+), ([+-]\d+)\]\)"
)


def results_tables() -> str:
    """One table per reports/eval_*.txt, showing games as they finish."""
    out = ""
    for path in sorted((REPO_ROOT / "reports").glob("eval_*.txt")):
        lines = path.read_text().splitlines()
        if not lines:
            continue
        rows = ""
        for line in lines[1:]:
            m = RESULT_LINE.match(line)
            if m:
                _, opp, w, d, lo, pct, diff, low, high = m.groups()
                rows += (
                    f"<tr><td>{html.escape(opp)}</td><td>{w} / {d} / {lo}</td><td>{pct}%</td>"
                    f"<td><b>{diff}</b> <span class='muted'>({low} to {high})</span></td></tr>"
                )
        title = html.escape(lines[0].strip("- "))
        out += (
            f"<div class='chart'><h3>{title}</h3><table class='res'><tr><th>Opponent</th>"
            f"<th>W / D / L</th><th>Score</th><th>Elo diff (95% range)</th></tr>{rows}</table></div>"
        )
    return out or "<p class='muted'>No evaluation results yet.</p>"


def running_jobs() -> str:
    names = {
        "uv run train": "Supervised training",
        "eval_net.py": "Strength evaluation",
        "selfplay": "Self-play training loop",
        "prepare_data.py": "Data preprocessing",
    }
    ps = subprocess.run(["ps", "-axo", "etime=,command="], capture_output=True, text=True).stdout
    jobs = []
    for line in ps.splitlines():
        if "/bin/zsh" in line or "pgrep" in line or "uv run" in line and "train" not in line:
            continue
        for key, label in names.items():
            if key in line and ".venv/bin/python" in line or key == "uv run train" and key in line:
                jobs.append(f"<li><b>{label}</b> <span class='muted'>running {line.split()[0]}</span></li>")
                break
    return "<ul>" + "".join(jobs) + "</ul>" if jobs else "<p class='muted'>Nothing running.</p>"


def recent_commits() -> str:
    log = subprocess.run(
        ["git", "log", "-8", "--pretty=format:%h|%ar|%s"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout
    rows = ""
    for line in log.splitlines():
        sha, when, subject = line.split("|", 2)
        rows += f"<tr><td><code>{sha}</code></td><td>{html.escape(subject)}</td><td class='muted'>{when}</td></tr>"
    return f"<table>{rows}</table>"


def stage_table() -> str:
    """Rating per stage from results/summary.csv (written by `uv run evaluate`)."""
    path = REPO_ROOT / "results" / "summary.csv"
    if not path.exists():
        return "<p class='muted'>No stage ratings yet (run uv run evaluate).</p>"
    rows = ""
    for line in path.read_text().splitlines()[1:]:
        stage, rating, low, high, games, *_ = line.split(",")
        rows += (
            f"<tr><td>{html.escape(stage)}</td><td><b>{float(rating):.0f}</b> "
            f"<span class='muted'>({float(low):.0f} to {float(high):.0f})</span></td>"
            f"<td>{games}</td></tr>"
        )
    return (
        "<table class='res'><tr><th>Stage</th><th>Elo, Stockfish scale (95%)</th>"
        f"<th>Games</th></tr>{rows}</table>"
    )


def selfplay_table() -> str:
    path = REPO_ROOT / "checkpoints" / "selfplay" / "iterations.jsonl"
    if not path.exists():
        return "<p class='muted'>Self-play has not run yet.</p>"
    rows = ""
    for line in path.read_text().splitlines():
        r = json.loads(line)
        verdict = "promoted" if r["promoted"] else "rejected"
        rows += (
            f"<tr><td>{r['iteration']}</td><td>{r['games']} games / {r['positions']:,} pos</td>"
            f"<td>{r['policy_loss']:.3f} / {r['value_loss']:.3f}</td>"
            f"<td>+{r['gate_wins']} ={r['gate_draws']} -{r['gate_losses']} "
            f"({r['gate_score']:.0%})</td><td>{verdict}</td></tr>"
        )
    return (
        "<table class='res'><tr><th>Iter</th><th>Self-play</th><th>Policy / value loss</th>"
        f"<th>Gate vs best</th><th>Result</th></tr>{rows}</table>"
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
        elif note.startswith("Built"):
            status, cls = "code ready", "run"
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
table.res{{border:0}} th{{text-align:left;font-size:12px;color:var(--muted);padding:6px 12px}}
ul{{margin:0;padding-left:18px}} code{{font-size:12px}}
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
<h3>Running right now</h3>
<div class="chart">{running_jobs()}</div>
<h3>Strength results (live)</h3>
<div class="charts">{results_tables()}</div>
<h3>Stage ratings (results/summary.csv)</h3>\n<div class="chart">{stage_table()}</div>\n<h3>Self-play iterations</h3>\n<div class="chart">{selfplay_table()}</div>\n<h3>Project steps</h3>
<table>{rows}</table>
<h3>Recent changes (git)</h3>
{recent_commits()}
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
