# Procedural image-to-Three.js miner

Pipeline: a reference image is analysed by a multimodal code model which emits a
`generate(THREE)` module; candidates are checked for conformance and the best is submitted.

## Run-level time governor

The validator sums each audit repeat's batch wall-clocks and rejects the whole audit above
`TOTAL_GENERATION_TIME_LIMIT` (7200 s). A per-batch cap cannot protect that ceiling, so
`pipeline.run_time_budget` governs the sum across a repeat's batches, sliced as
`remaining / batches_left` and floored at `min_batch_budget`. Counters reset at a repeat
boundary because each repeat is timed separately.

Truncating a batch marks its unfinished prompts failed, and more than
`MAX_MISMATCHED_PROMPTS` failures rejects the audit just as surely as running long — so the
governor sheds *work* rather than time: `adaptive_ensemble` scales the candidate count by
the previous batch's completion rate (elapsed time cannot reveal over-subscription, since a
truncated batch always reports exactly its budget).

## Hardware

`hardware.json` declares `4xH200`.

## Ensemble and sampling

`actors.coder.ensemble_size: 60` with `ensemble_temperature: 0.3`.

The temperature is deliberately below the common 0.6. Measured on this coder across the
same prompts and seeds at K=30, t=0.3 gives sd **0.1133** against t=0.6's **0.1534** — 26%
less spread — while the expected best-of-K is unchanged (+0.002). Candidate diversity does
not come from temperature here: every draw is already a distinct program (360/360 unique at
K=120, and 90/90 unique even at t=0.3), so temperature is free to be tuned for stability
rather than variety.

## GPU split

Coder on GPUs 0-2 (`data_parallel_size: 3`), judge/critic on GPU 3. This is intentional and
not an idle card: the bracket runs (K-1) duels per prompt, so the judge is close to fully
utilised on its own GPU for the whole run. Moving a fourth coder replica onto GPU 3 speeds
the coder up but makes the judge the critical path, which pushes the run past the
generation-time limit.
