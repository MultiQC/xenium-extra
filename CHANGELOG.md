# multiqc-xenium-extra changelog

## v1.1.0 [2026-05-12]

- Add new "Nuclei per Cell" stacked bar plot showing the per-sample
  distribution of cells with 0, 1, or 2+ segmented nuclei.
- Add hidden general-stats columns "% 0-Nuclei Cells" and
  "% Multi-Nuclei Cells".
- **Fix single-sample density plots.** On single-sample reports, three
  cell-related density sections — "Cell Area Distribution",
  "Nucleus to Cell Area", and the transcripts-per-cell side of
  "Distribution of Transcripts/Genes per Cell" — silently skipped
  because the parser stored only summary statistics (`*_box_stats`)
  and the single-sample density helpers require raw per-cell value
  lists. `parse_cells_parquet` now emits `cell_area_values`,
  `nucleus_to_cell_area_ratio_values`, and `total_counts_values`
  alongside the existing box-stats summaries, so all three sections
  render as KDE density plots on single-sample reports (matching the
  helptext's existing description). Multi-sample reports are
  unaffected. Note: `multiqc_data.json` is now larger by `O(n_cells)`
  values per added metric per sample (~12 MB for a 500k-cell sample).
- **Fix "Fraction of Transcripts in Nucleus" plot.** Previous releases
  computed this incorrectly as `nucleus_count / total_counts` from
  `cells.parquet`, where `nucleus_count` is the number of segmented
  nuclei per cell (typically 0/1/2+), not a transcript count. The plot
  is now correctly derived from `transcripts.parquet` via the
  per-transcript `overlaps_nucleus` flag.

  **Note for users tracking these values historically**: post-fix
  values will be substantially different from pre-1.1.0 values
  (approximately 6× larger on representative samples). The previous
  numbers were not biologically meaningful; the new values are.
  Downstream analyses that consumed `nucleus_rna_fraction_mean` /
  `nucleus_rna_fraction_median` from `multiqc_data.json` should be
  re-baselined.

## v1.0.2 [2025-12-10]

Increase file size limit from 5GB to 50GB to handle larger Xenium files.

## v1.0.1 [2025-10-25]

Move over some additional code from core MultiQC that
was missed in the initial migration.

## v1.0.0 [2025-10-25]

Initial release of the `multiqc-xenium-extra` plugin.
Removes much of the code and dependencies from core MultiQC
into an optional add-on plugin.

Does not affect report content if both plugin and MultiQC are installed.
