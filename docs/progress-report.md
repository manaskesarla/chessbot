# Progress report: what's built, with what, and why

Status: Steps 0 to 5 are done. Steps 6 to 10 are not started. 101 automated tests pass, and ruff and mypy report no issues.

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

## What's next

| Step | Goal |
|---|---|
| 6: MCTS search | Let the network "calculate" by exploring move sequences. Expected to add several hundred Elo over the 1300 starting point. |
| 7: Self-play RL | The bot plays itself with search, learns from the results, and only accepts a new version if it beats the old one. |
| 8: Rigorous evaluation | At least 100 games per level, with an Elo-versus-stage chart. |
| 9: Lichess bot | A live public rating (needs a fresh Lichess account from you). |
| 10: Resume packaging | Final README, ablations, honest write-up. |

**Honest outlook:** on a Mac with no cloud spend, ~1750 on Lichess is a stretch. The plan is to report whatever is measured, with the contribution of each stage.
