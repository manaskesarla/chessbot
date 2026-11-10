# Progress report: what's built, with what, and why

Status: Steps 0 to 7 are done and Step 8 is built and partly run. Steps 9 and 10 are not started. 132 automated tests pass. Step 7 finished with a measured result: self-play gave no statistically clear gain over the supervised network.

**Technology used across the project**

| Tool | What it does here |
|---|---|
| Python 3.11 + `uv` | The language, and the tool that installs dependencies and runs commands |
| PyTorch (`mps`) | Builds and trains the neural network, running on the Mac's GPU |
| python-chess | Chess rules, legal moves, PGN game files, and talking to Stockfish |
| Stockfish 19 | A strong open-source engine, weakened on purpose to act as a measuring stick |
| NumPy | Fast array maths for data and encoding |
| zstandard | Reads the compressed Lichess game files without unpacking them to disk |
| YAML + dataclasses | Settings live in `configs/*.yaml` instead of in code |
| pytest, ruff, mypy | Tests, style checks, and type checks |
| Git / GitHub | Version history, pushed to `manaskesarla/chessbot` |

---

## Step 0: Success and measurement

**What was done:** Wrote `docs/evaluation-protocol.md`. It fixes how every strength number in the project is measured.

**Concepts**
- **Elo rating:** a number for playing strength, estimated from game results. Ratings on different sites are not comparable, so the protocol states "Lichess rapid" as the target scale.
- **Fair comparison:** every comparison uses a fixed thinking budget per move, alternates colours, and starts from varied openings. This stops a bot from looking stronger just because it thought longer or got easier openings.
- **Confidence interval:** the range the true Elo probably lies in. The protocol asks for at least 100 games per opponent for any number that gets reported.
- **Fallback claim:** the protocol decides in advance what to claim if 1750 isn't reached. The point is to avoid moving the goalposts later.

---

## Step 1: Environment and tooling

**What was done:**
- Created the project (`pyproject.toml`), the folder layout (`src/chessbot/...`, `configs/`, `tests/`, `scripts/`), and typed YAML config loading. Misspelled setting names are rejected instead of silently ignored.
- Installed Stockfish with Homebrew.
- Added `uv run smoke`, which checks the GPU, python-chess, and Stockfish in one go.

**Concepts**
- **Virtual environment:** project-only packages, so nothing collides with the rest of your Mac.
- **MPS:** Apple's GPU backend for PyTorch. It is much faster than the CPU for neural networks.
- **Configs as files:** changing an experiment means editing a YAML file rather than code, which keeps runs reproducible.
- **Tests and type checks from day one:** they catch mistakes early, which matters a lot when bugs can be silent.

---

## Step 2: State and move representation

**What was done:**
- `encode_board`: turns any position into 19 layers of 8x8 numbers.
- `move_to_index` / `index_to_move`: map every move to one of 4,672 slots, and back.
- `legal_mask`: marks which of the 4,672 slots are legal in a position.
- Tests: every legal move round-trips across more than 10,000 positions. Castling, en passant, promotions (including underpromotions), and the flip for Black are checked. Encoding runs at about 55,000 positions per second.

**Concepts**
- **A neural network needs numbers, not a board.** The 19 layers are 12 for pieces (6 types for each side), 4 for castling rights, 1 for en passant, 1 for the halfmove clock, and 1 for whose turn it is.
- **Side-to-move perspective:** the board is flipped when Black is to move, so the network only ever learns one point of view, "my pieces move up the board".
- **Action space (8x8x73):** 64 starting squares times 73 move types (56 sliding moves, 8 knight moves, 9 underpromotions). That is the AlphaZero scheme.
- **Masking:** illegal moves are forced to zero probability, so the bot can never choose one.
- **Why test so heavily:** an encoding mistake doesn't crash anything. The network just quietly learns nonsense.

---

## Step 3: Baselines

**What was done:**
- Three reference bots: random, greedy (picks the best material after one move), and alpha-beta search (depths 2 and 3).
- A match runner that plays any bot against any other, or against Stockfish limited to a chosen Elo.
- An opening book of 20 varied openings, each played once with each colour.
- Elo estimates with 95% confidence intervals.
- A bug caught by the tests: the alpha-beta bot treated moves it had skipped as tied for best, so it sometimes played bad moves. Fixed.
- Quick check result, 10 games per matchup: random < greedy < alpha-beta depth 2 < depth 3. Depth 3 scored 9/10 against Stockfish 1320.

**Concepts**
- **Baseline:** a weak, simple reference. If the learned bot can't beat a hand-written search, the learning isn't adding value.
- **Minimax and alpha-beta pruning:** the bot assumes both sides play their best, and skips branches that cannot change the answer.
- **Quiescence search:** keeps searching captures at the end of the search, so the bot doesn't stop mid-trade and misjudge a position.
- **Piece-square tables:** each piece gets a bonus or penalty for the square it stands on, for example knights prefer the centre.
- **Elo from results:** a score of 50% means equal strength, and 75% is about +190 Elo. The confidence interval shrinks as the number of games grows.

---

## Step 4: Network architecture

**What was done:** A two-headed residual network with 6 blocks and 64 filters, 505,228 parameters, plus checkpoint saving and loading that stores the model settings with the weights. It processes about 27,000 positions per second on the Mac GPU at batch size 64.

**Concepts**
- **Convolutional network:** small filters slide over the board to detect patterns such as pins or pawn chains.
- **Residual (skip) connections:** each block adds its input to its output, which makes deeper networks train reliably.
- **Policy head:** outputs a score for each of the 4,672 moves, meaning "which moves look good".
- **Value head:** outputs one number from -1 (losing) to +1 (winning), meaning "who is ahead".
- **Why small:** search will call the network thousands of times per move, so speed matters more than size.
- **Batching:** the GPU is far more efficient on many positions at once (about 800 positions per second one at a time, about 27,000 in batches). This drives the search design in Step 6.

---

## Step 5: Supervised pretraining

**What was done:**
- Downloaded Lichess's June 2016 games (1.09 GB, compressed) and streamed through them (3.1 million games scanned).
- Kept 150,000 games where both players were rated 2000 or higher. That gave **10,750,564 positions**.
- Stored each position compactly (about 70 bytes instead of 5 KB), then expanded to the 19 layers on the fly. A test proves the fast path matches the Step 2 encoder exactly.
- Trained for about 82,000 steps (2 passes over the data), learning the human's move and the game result together.
- Held out 2% of games, split by game so no game appears in both training and testing.

**Results on held-out games:** top-1 move accuracy **48.1%**, top-5 accuracy **86.9%**, value error 0.72.

**Strength of the raw network (no search), 20 games per opponent:**

| Opponent | W / D / L | Elo difference (95% range) |
|---|---|---|
| random | 18 / 2 / 0 | +512 (+353 to +636) |
| greedy | 16 / 4 / 0 | +382 (+255 to +636) |
| alpha-beta depth 2 | 7 / 5 / 8 | -17 (-159 to +119) |
| alpha-beta depth 3 | 0 / 9 / 11 | -215 (-353 to -120) |
| Stockfish 1320 | 7 / 6 / 7 | 0 (-134 to +134) |
| Stockfish 1500 | 2 / 3 / 15 | -269 (-594 to -132) |

That puts the raw network at roughly **1300 on Stockfish's scale**. It is a rough number from few games, and it is not a Lichess rating.

**Concepts**
- **Supervised learning:** learn by imitating examples. Here the examples are strong humans' moves and results.
- **Why start here:** without it, self-play would spend days rediscovering how pieces move.
- **Cross-entropy loss (policy):** rewards the network for giving high probability to the move the human played.
- **Mean squared error (value):** penalises wrong guesses about who won.
- **AdamW, warm-up, and cosine decay:** the optimiser and learning-rate schedule (start gently, then taper), which keep training stable.
- **Held-out validation by game:** measures real learning rather than memorisation.
- **Top-1 / top-5 accuracy:** how often the human's move is the network's first choice or among its five best.
- **Resumable training:** progress is saved periodically, so an interruption doesn't lose hours.

---

## Step 6: MCTS search

**What was done:**
- Batched PUCT Monte Carlo Tree Search with virtual loss, first-play urgency (0.2), Dirichlet noise at the root, exact values for checkmate/draw positions, and a position cache so repeated positions aren't re-evaluated.
- `MCTSBot` (network plus search) and a heuristic evaluator used to test the search separately from the network.
- `scripts/benchmark_mcts.py` and search support in `scripts/eval_net.py`. Search runs at about 1,000 simulations per second, and the mate-in-N puzzle tests pass.
- Defaults in `configs/mcts.yaml`: 100 simulations, c_puct 1.5, batch size 16.

**Strength with 100 simulations per move, 40 games per opponent (same network as Step 5):**

| Opponent | W / D / L | Elo difference (95% range) |
|---|---|---|
| random | 40 / 0 / 0 | +759 |
| greedy | 40 / 0 / 0 | +759 |
| alpha-beta depth 2 | 30 / 6 / 4 | +269 (+167 to +440) |
| alpha-beta depth 3 | 16 / 13 / 11 | +44 (-44 to +138) |
| Stockfish 1320 | 30 / 3 / 7 | +228 (+121 to +396) |
| Stockfish 1500 | 22 / 5 / 13 | +80 (-20 to +194) |
| Stockfish 1700 | 10 / 11 / 19 | -80 (-181 to +10) |

Search lifts the same network from roughly 1300 to the high 1500s or 1600s on Stockfish's scale.

**Concepts**
- **MCTS:** instead of trusting the network's first guess, the bot tries many move sequences and favours the ones that keep looking good.
- **PUCT:** the rule for which move to explore next, balancing the network's prior, the results so far, and how rarely a move has been tried.
- **Virtual loss and batching:** several searches run at once and are sent to the GPU together, which is what makes the network fast enough to use inside search.
- **Dirichlet noise:** random extra exploration at the root, used in self-play so games don't repeat.

---

## Step 7: Self-play reinforcement learning

**What was done:**
- `src/chessbot/selfplay/`: parallel game generation (4 workers), a replay buffer, and the loop in `loop.py`. Each iteration plays 64 games with 64 simulations per move, then trains a candidate network on the stored search results, mixed 50/50 with human positions to stop it drifting away from sensible chess. The candidate then plays 32 gate games against the current best, and replaces it only with a score of at least 55%.
- The loop saves its state after every iteration, so an interrupted run resumes where it stopped.
- **Small first run (8 iterations of 64 games, ~12 min):** a smoke test. Only iteration 2 was promoted (57.8% on 32 games, within noise). Its network rated 1646 (1607 to 1687).
- **Larger run (12 iterations, ~2 hours, `configs/selfplay.yaml`):** 200 games per iteration at 128 simulations per move, 800 training steps, a 300,000-position buffer, and 100 gate games at 128 simulations. Each iteration took about 10.5 minutes (games 4 min, training 1 min, gating 5 min). It started fresh from the supervised network.

| Iter | Gate score | Result |
|---|---|---|
| 1 | 51.0% | rejected |
| 2 | 50.0% | rejected |
| 3 | 52.5% | rejected |
| 4 | 39.5% | rejected |
| 5 | 48.0% | rejected |
| 6 | 48.0% | rejected |
| 7 | 48.5% | rejected |
| 8 | 41.0% | rejected |
| 9 | 55.0% | **promoted** |
| 10 | 43.5% | rejected |
| 11 | 41.5% | rejected |
| 12 | 54.5% | rejected |

**Evaluation (MCTS with 100 simulations, 400 games each: 100 per Stockfish level at 1320, 1500, 1700, 1900):**

| Network | Rating (Stockfish scale) | 95% interval |
|---|---|---|
| Supervised + MCTS | 1630 | 1591 to 1667 |
| Small self-play run (iteration 2) + MCTS | 1646 | 1607 to 1687 |
| Larger self-play run (iteration 9) + MCTS | **1647** | 1611 to 1687 |

Per level for the larger run (W / D / L): 1320: 82/4/14, 1500: 54/14/32, 1700: 39/17/44, 1900: 12/27/61.

**The three "done" checks:**
- Runs unattended and resumes after interruption: met.
- At least some iterations promote: met, but weakly. One of 12 promoted, and at exactly the 55% threshold.
- Final net beats supervised + MCTS: higher by about 17 Elo, but the intervals overlap almost completely, so this is not a real gain.

**What this means:** the self-play loop works end to end and the results are measured honestly, but at this scale (about 20,000 new positions per iteration against 10.75 million human ones) it did not clearly improve the network. The resume claim should be "built and evaluated a gated self-play loop; no significant gain at laptop scale", not "+X Elo from RL". Meaningful gains would likely need far more games and iterations than a Mac can run in a few hours.

**Concepts**
- **Self-play:** the bot generates its own training data by playing itself, with search as the "teacher".
- **Policy target:** the search's visit counts, which are usually better than the raw network's guess, so the network learns to predict them.
- **Replay buffer:** recent positions are kept and sampled at random, so training isn't dominated by the latest games.
- **Gating:** a new network is only accepted if it beats the current one, which protects against a bad iteration making things worse.

---

## Step 8: Rigorous evaluation

**What was done:**
- `src/chessbot/eval/`: a harness that plays a network against Stockfish at several fixed Elo levels (1320, 1500, 1700, 1900 in `configs/eval.yaml`), a bootstrap fit that turns all the results into one rating with a 95% interval, and an Elo-by-stage chart. Run with `uv run evaluate`, with `--results-dir` for experiments.
- Results are written to `results/<experiment>/summary.csv` and `elo_by_stage.png`, and shown on the dashboard.
- **Search-budget experiment (`results/sims_scaling`):** MCTS at 100 simulations scored 1696 (1613 to 1789) on 60 games against Stockfish 1500 and 1700. The 400-simulation run was started but did not finish.
- **Full-protocol runs (400 games each, results in `results/summary.csv` and `results/elo_by_stage.png`, committed):** supervised + MCTS 1630 (1591 to 1667), and the two self-play networks at 1646 and 1647. See Step 7 for the table.
- **Still to run:** the earlier stages under the same protocol (random, greedy, alpha-beta, raw network without search) so the chart covers every stage.

**Concepts**
- **Stockfish scale:** the rating is relative to a weakened Stockfish at known Elo levels. It is not a Lichess rating.
- **Bootstrap:** resampling the game results many times to see how much the rating would wobble, which gives an honest interval.

---

## Other tooling

- `scripts/dashboard.py` writes `reports/progress.html`: training curves, running jobs, live log tails, Stockfish results, a self-play progress bar (percentage, iteration count and current phase in plain words), the self-play iteration table, and step status. It refreshes every 10 seconds with `--watch`.

---

## What's next

| Step | Goal |
|---|---|
| 8 (finish) | Run the earlier stages (random, greedy, alpha-beta, raw network) under the same 400-game protocol so the Elo-by-stage chart is complete. |
| 9: Lichess bot | A live public rating (needs a fresh Lichess account from you). |
| 10: Resume packaging | Final README, ablations, honest write-up. |

**Honest outlook:** supervised network plus search rates about 1630 to 1650 on the Stockfish scale, which is encouraging, but Lichess ratings differ and self-play has not shown a gain. The plan is to report whatever is measured, with the contribution of each stage.
