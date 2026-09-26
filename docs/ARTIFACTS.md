# Artifact guide

`data-aggregates/` holds the aggregate analysis JSONs the paper cites.
Per-fact margin fields are the three margin tarballs on the accompanying
data record (`margins-campaign1.tar.gz`, `margins-campaign1-ext.tar.gz`,
`margins-campaign2.tar.gz`), enumerated with sha256 hashes in
`manifest.sha256.json`.

Layout of each margin shard: one JSON line per (fact, battery) with
`rank_bar`, `m_top1`, `m_nll`, `logp`, and the alive flag under the model's
own ghost-Q99 threshold; the `__meta__` line carries the model id and the
ghost quantile thresholds.

Also in `data-aggregates/`:

- `truegen-shards/` — the per-probe free-generation outputs behind the
  candidate-free readout (30 gzipped JSONL shards, one per model per pair).
- `e6main-qa-dumps/` — per-fact QA dumps for the 1B bridge units
  (joint endpoint-by-merge decomposition).
- `e6scan-reports/` — the per-variant reports of the 1B injection recipe scan.
- `truemerge_report.json`, `margins_tier2.skipped.json` — true-merge arm
  report and the tier-2 skip ledger.

Every figure in the paper regenerates from these artifacts via the scripts
in `code/` (see the figure-provenance appendix of the paper for the mapping).

## Path mapping

The paper text cites artifacts by their analysis-tree paths. In this
package they live at: `aug/<name>.py` -> `code/aug/<name>.py`;
`figures/src/<name>.py` -> `code/figures/src/<name>.py`;
`aug/<name>.json` -> `data-aggregates/<name>.json`;
`results_aug_truegen/shards/` -> `data-aggregates/truegen-shards/`;
`results_aug_e6main*/qa_dump.jsonl` -> `data-aggregates/e6main-qa-dumps/`;
`results_aug_e6scan_*/e6main_report.json` -> `data-aggregates/e6scan-reports/`.
