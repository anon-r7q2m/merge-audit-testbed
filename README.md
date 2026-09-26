# Merge-audit testbed — released artifacts

Anonymous companion repository for the paper's reproducibility statement.

## Contents

- `docs/PROTOCOL.md` — the analysis protocol as specified before data
  collection, plus the pre-registered/post-review evidence timeline.
- `docs/ARTIFACTS.md` — guide to the data aggregates and margin shards.
- `code/` — the training/evaluation drivers and analysis scripts for every
  arm in the paper (testbed grid, campaigns, allocation sweep, scale ladder
  and recipe controls, 1B bridge, true-merge arm, free-generation readout,
  null analyses, figure renderers).
- `data-aggregates/` — the aggregate analysis JSONs cited in the paper.
- `manifest.sha256.json` — sha256 manifest of the released margin fields
  (849 files over the three campaigns/units).

The per-fact margin fields themselves are the three data tarballs
accompanying this record (see `docs/ARTIFACTS.md` for their layout).

## License

CC-BY-4.0 for text and data aggregates; MIT for code.
