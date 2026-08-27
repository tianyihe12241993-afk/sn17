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
