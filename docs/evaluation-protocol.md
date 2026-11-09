# Evaluation protocol

Every strength number in this project is measured this way, so results are comparable across stages.

## Scale
"Rating" means **Lichess rapid**, read from the live bot account (Step 9). Before the bot is
online, strength is estimated against Stockfish with `UCI_LimitStrength` on. Stockfish's
`UCI_Elo` scale is its own calibration, not Lichess, so those numbers are labelled
"vs Stockfish scale" and are never quoted as a Lichess rating.

## Fairness rules
- Fixed budget per move, never mixed within one comparison:
  - Stockfish: 50 ms per move (`chessbot.eval.match.StockfishBot`)
  - Our network + search: a fixed number of MCTS simulations (`configs/eval.yaml`)
  - Alpha-beta baselines: fixed depth
- Colours alternate; each opening from `chessbot.eval.openings` is played once with each colour.
- Games over 400 plies are adjudicated as draws. Draws by repetition and 50 moves are claimed.
- Hardware: the same MacBook, nothing else heavy running.

## Opponents and size
- Stockfish `UCI_Elo` levels: 1320, 1500, 1700, 1900 (the lowest the engine accepts is 1320).
- At least **100 games per level** for any number that goes into the README or resume.
  Quick checks during development may use fewer, and are not reported.
- Choose levels within a few hundred Elo of the bot; very one-sided matches carry little information.

## Estimating Elo
- Score = (wins + draws/2) / games.
- Elo difference = -400 * log10(1/score - 1), plus the opponent's `UCI_Elo`.
- 95% interval from the standard error of the per-game scores (`chessbot.eval.elo`).
- Scores of 0% or 100% are clamped; treat those intervals as "play a different opponent".

## Reported stages
random, greedy, alpha-beta, supervised network (no search), supervised + MCTS, then each
promoted self-play iteration. Results go to `results/` as CSV/JSON.

## Fallback claim (decided in advance)
- Reach ~1750 Lichess rapid: state it with the rating and game count.
- Fall short: state the measured Lichess rating and the Elo gained from each stage
  (search, RL). A lower number with clean ablations is the deliverable.
