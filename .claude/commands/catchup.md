---
description: Get up to speed on CustomRAFT with as few tokens as possible
---
Catch up on this project cheaply. Do exactly this, nothing more:

1. Read `kb/INDEX.md` (start with its "Now" section).
2. Run `git log --oneline -8` to see what changed recently.
3. Check whether training is running: find the newest `train.log` under `experiments/*/runs/*/` (by modification time) and show its last line, with how many minutes ago it was modified (a log updated within the last 5 minutes means a run is active).

Do NOT open `kb/legacy/`, `analysis/` files, checkpoints or other cards unless the user asks.

Then report in at most 10 lines:
- Where things stand (last experiment, what is known, from INDEX)
- What is pending (from "Now", open questions, undecided reviews)
- What is running (or "nothing training")
- If anything in the git log or the run status contradicts the "Now" section, say so.

End by asking what the user wants to work on. Follow the KB rules in CLAUDE.md: KB content is drafted in chat and written only after the user approves it.
