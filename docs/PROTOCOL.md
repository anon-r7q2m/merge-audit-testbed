# Analysis protocol (released with the margin fields)

This document records the analysis protocol of the accompanying paper in
neutral terms, as specified before data collection, plus the evidence
timeline separating pre-registered from post-review arms.

## Universe (synthetic testbed)

- Generator seed 20260814, held fixed across runs; vocabulary 8,192 with a
  dedicated 128-value segment per attribute; facts are (entity, attribute,
  value) triples, the value always a single sentence-final token.
- Pools: 6,000 shared facts, 6,000 exclusive facts per branch, 2,000 ghost
  facts (generated, never rendered), 200,000 filler entities. Exclusive sets
  are mutually exclusive and pairwise twin-balanced (`balance_pass = true`
  at exposure count K=128).
- Probes: 8 training paraphrase templates per attribute (storage-layer
  battery) plus 3 held-out templates (readout-layer battery). The universe
  regenerates deterministically; the two training campaigns' fact files were
  verified byte-identical after decompression.

## Training

- 29.6M-parameter decoder-only GPT (8 layers, d=512, d_ff=2048, context 512,
  tied embeddings), AdamW, bf16.
- Base segment: 300M tokens of filler plus shared facts. Branches A/B fork
  from the same base weights, inheriting its Adam moments; each learns its
  6,000 exclusive facts entirely inside the first post-fork exposure block at
  final dose K=128 (25M tokens; the protocol's dose ladder moved K from 64 to
  128 after the storage-health check at K=64 returned frac_rank1
  0.8913/0.9090 against the 0.90 bar, a pre-specified mechanical branch).
  Fact:filler ratio ~1:1 with identical slot patterns. Learning rate constant
  3e-4 after warmup.
- Drift axis tau: filler tokens after the exposure block, grid
  {0, 3.1M, 6.3M, 12.5M, 25M, 50M, 100M, 200M}; nested design (one deep run
  per seed, mid-run checkpoints as shallower branches).
- Two campaigns of three runs each: seeds {42, 1042, 2042} (campaign 1) and
  seeds {3184, 5383, 7192} (campaign 2). Campaign 2 is isomorphic with one
  declared difference: its storage-health gate passed at K=64, the
  pre-authorized ladder endpoint, so its exposure block is 12.5M tokens.
  Campaign-2 readouts are counted alongside campaign-1 readouts, never
  averaged into a single point estimate.
- A later campaign-1 extension unit (seeds {6784, 403, 1189}, same protocol,
  K pinned at 128; storage health 0.948-0.966 on all six new endpoint
  evaluations) adds three runs: the paper counts n = 9.

## Protocol health checks

- Micro-benchmark: measured 292,397 tok/s against an assumed 261,111
  (ratio 1.12, within the 2x bar).
- Storage-health check at tau=0: storage-battery frac_rank1 0.9278/0.9443 on
  the gate seed, both branches above 0.90.
- Spawn check: the tau=0 merge-point filler loss sits 13.6% below the
  endpoint mean (relative barrier -0.1357 against a +0.30 alert bar).

## Readouts

Per (fact, template), with z the last-position logits restricted to the
1,024 value tokens: rank = 1 + #{v != v*: z_v > z_v*}; the main caliber is
the raw-logit top-1 competitor margin m_top1 = z_v* - max rest; the secondary
caliber is the NLL margin. Per-fact aggregation is the template mean; the
rank-1 criterion is rank <= 1.5; the survival threshold is per-model adaptive
(Q99 of the model's own ghost-fact storage-layer margins).

## Storage gates

Arm-specific, declared with each arm: the K-ladder bar above (universe 1);
>= 0.85 frac-rank-1 on the scale ladder; endpoint QA accuracy on the 1B
bridge.

## Evidence timeline (what was pre-registered vs added post-review)

- **Pre-registered (frozen before data collection)**: the universe, training
  protocol, dose ladder with its mechanical storage-health branch, the
  tau grid, the readout calibers and thresholds, the three-line residual
  decomposition, the statistical-unit convention (the training run), and the
  primary comparisons (campaign-1 grid).
- **Post-review arms (declared as such in the paper)**: campaign 2, the
  campaign-1 extension unit, universe 2 and the fresh campaign, the scale
  ladder and its recipe-ablation controls, the allocation sweep and its 1B
  replication, the true-merge arm (five real pairs) and its free-generation
  readout, the 1B by-construction bridge and its storage-margin scan, the
  operator arms (TIES/DARE), the damage-machine power check, and the
  instrument self-audit extensions (quantile sweep, null-reference test,
  NLL-barrier placebo).
