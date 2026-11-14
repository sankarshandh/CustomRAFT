# Questions

Open questions, one row each. ID = origin + number: `E003-Q1` came out of E003, while `G-Q1` is general (not tied to one experiment).
A question is closed when something answers it: set the status to `answered → <finding / study / experiment id>`.

| id | question | raised (date, by) | related | status |
|---|---|---|---|---|
| E003-Q1 | Do biphasic units improve flow if they **can't** be silenced (spatial route closed too)? | 2026-11-12, Claude | E003-F1 | open → backlog #1 |
| E003-Q2 | **Why** does training silence transient units? Hypothesis: the loss only reads features through correlation matching, and transient responses match less consistently between the two encoder windows. | 2026-11-12, Claude | E003-F1; legacy biphasic-collapse note (H1) | open |
| E003-Q3 | Is the result the same with another random seed? (E003 is a single run.) | 2026-11-12, Claude | E003-F1 | open |
| E003-Q4 | Would neuron 1's surviving motion signal be used if it reached the flow update directly, not only through correlation? | 2026-11-12, Claude | E003-F1, S001 | open |
| E004-Q1 | Can fine-tuning **revive** biphasic units that E003 already switched off, or is that state a trap? In E003 each biphasic unit was switched off by making its centre and surround the same width, so its kernel is about zero and its output is about zero for every input. Training changes a parameter only as far as the change would alter the loss. A unit whose output is zero (and whose readers downstream have learned to ignore it) barely alters the loss whichever way its widths move, so the widths stay put, like a "dead" ReLU. If so, E004's lack of change says nothing about whether asymmetric views work; it only says fine-tuning from E003 can't get there. Test: (a) at fine-tune start, reset the biphasic units' widths to their initial values (new train.py argument, default off), or (b) train from scratch with `asym_prob` 0.3 (candidate E005). | 2026-11-13, Claude | E004-R1, E004 card, E003-F1 | answered → [E004-F2](findings/E004-F2-silencing-not-a-finetune-trap.md): not a fine-tuning trap; from scratch they are switched off too |
