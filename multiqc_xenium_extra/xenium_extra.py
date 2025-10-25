"""MultiQC Xenium Extra plugin functions"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import importlib_metadata
import numpy as np
import polars as pl
from scipy import stats
import scanpy as sc

from multiqc import config
from multiqc.plots import bargraph, box, linegraph, scatter, table
from multiqc.plots.table_object import ColumnDict, TableConfig
from multiqc.utils import mqc_colour
from multiqc.modules.xenium.xenium import GENE_CATS, categorize_feature

log = logging.getLogger("multiqc")



def xenium_extra_execution_start():
    """Code to execute after config files and command line flags have been parsed.

    This hook adds search patterns for large Xenium files and adjusts file size limits.
    """

    # Get plugin version
    version = importlib_metadata.version("multiqc_xenium_extra")
    log.debug(f"Running MultiQC Xenium Extra Plugin v{version}")

    # Add search patterns for parquet and H5 files
    # These are the computationally intensive files that the plugin processes
    # Note: shared=True allows multiple modules to access these files
    # Note: We don't check contents since these are binary files
    if "xenium/transcripts" not in config.sp:
        config.update_dict(
            config.sp,
            {
                "xenium/transcripts": {
                    "fn": "transcripts.parquet"
                }
            }
        )

    if "xenium/cells" not in config.sp:
        config.update_dict(
            config.sp,
            {
                "xenium/cells": {
                    "fn": "cells.parquet"
                }
            }
        )

    if "xenium/cell_feature_matrix" not in config.sp:
        config.update_dict(
            config.sp,
            {
                "xenium/cell_feature_matrix": {
                    "fn": "cell_feature_matrix.h5"
                }
            }
        )

    # Increase file size limit to handle large Xenium files (if not already set by user)
    # Default is 50MB, Xenium files can be several GB
    if config.log_filesize_limit < 5000000000:  # 5GB
        log.debug("Increasing log_filesize_limit to 5GB for Xenium parquet files")
        config.log_filesize_limit = 5000000000


# Extension function and helper functions below
def extend_xenium_module(xenium_module):
    """
    Extend a xenium module instance with extra parquet and H5 file analysis.

    This function:
    1. Parses parquet and H5 files
    2. Augments data_by_sample dict with cell area metrics
    3. Adds additional sections to the xenium module

    Sections added:
    - Transcript quality analysis
    - Field of view quality
    - Cell distributions
    - Cell area distributions
    - Nucleus RNA fraction
    - Nucleus-to-cell area ratios

    Args:
        xenium_module: Instance of the core Xenium MultiQC module
    """
    log.info("Extending Xenium module with extra analysis...")

    # Parse parquet files
    transcript_data_by_sample = {}
    cells_data_by_sample = {}

    # Parse transcripts.parquet files
    transcript_files = list(xenium_module.find_log_files("xenium/transcripts", filecontents=False, filehandles=False))
    log.debug(f"Found {len(transcript_files)} transcript parquet files")
    for f in transcript_files:
        parsed_data = parse_transcripts_parquet(f)
        if parsed_data:
            parent_dir = Path(f["root"]).name if f["root"] else f["s_name"]
            transcript_data_by_sample[parent_dir] = parsed_data
            xenium_module.add_data_source(f, parent_dir)

    # Parse cells.parquet files
    for f in xenium_module.find_log_files("xenium/cells", filecontents=False, filehandles=False):
        parsed_data = parse_cells_parquet(f)
        if parsed_data:
            parent_dir = Path(f["root"]).name if f["root"] else f["s_name"]
            # Merge with existing data if available
            if parent_dir in cells_data_by_sample:
                cells_data_by_sample[parent_dir].update(parsed_data)
            else:
                cells_data_by_sample[parent_dir] = parsed_data
            xenium_module.add_data_source(f, parent_dir)

    # Parse cell_feature_matrix.h5 files
    for f in xenium_module.find_log_files("xenium/cell_feature_matrix", filecontents=False, filehandles=False):
        parsed_data = parse_cell_feature_matrix_h5(f)
        if parsed_data:
            parent_dir = Path(f["root"]).name if f["root"] else f["s_name"]
            # Merge with existing cell data
            if parent_dir in cells_data_by_sample:
                cells_data_by_sample[parent_dir].update(parsed_data)
            else:
                cells_data_by_sample[parent_dir] = parsed_data
            xenium_module.add_data_source(f, parent_dir)

    # Filter out empty datasets
    transcript_data_by_sample = {k: v for k, v in transcript_data_by_sample.items() if v}
    cells_data_by_sample = {k: v for k, v in cells_data_by_sample.items() if v}

    if not transcript_data_by_sample and not cells_data_by_sample:
        log.info("No extra Xenium data found to extend module")
        return

    log.info(f"Found {len(transcript_data_by_sample)} samples with transcript data")
    log.info(f"Found {len(cells_data_by_sample)} samples with cell data")

    # Merge cell area metrics into data_by_sample for general stats
    for sample_name, cell_data in cells_data_by_sample.items():
        if sample_name in xenium_module.data_by_sample:
            # Add cell area metrics to existing sample data
            if "cell_area_median" in cell_data:
                xenium_module.data_by_sample[sample_name]["cell_area_median"] = cell_data["cell_area_median"]
            if "nucleus_area_median" in cell_data:
                xenium_module.data_by_sample[sample_name]["nucleus_area_median"] = cell_data["nucleus_area_median"]
            if "nucleus_to_cell_area_ratio_median" in cell_data:
                xenium_module.data_by_sample[sample_name]["nucleus_to_cell_area_ratio_median"] = cell_data[
                    "nucleus_to_cell_area_ratio_median"
                ]

    # Use transcript count from parquet file if missing from JSON
    for sample_name, transcript_data in transcript_data_by_sample.items():
        if sample_name in xenium_module.data_by_sample:
            # Add transcript count if missing from JSON data
            if (
                "num_transcripts" not in xenium_module.data_by_sample[sample_name]
                or xenium_module.data_by_sample[sample_name]["num_transcripts"] is None
            ):
                if "total_transcripts" in transcript_data:
                    xenium_module.data_by_sample[sample_name]["num_transcripts"] = transcript_data["total_transcripts"]

    # Add additional cell/transcript metrics to general stats
    # Core module already adds: num_transcripts, num_cells_detected, fraction_transcripts_assigned, median_genes_per_cell
    # Plugin adds: fraction_transcripts_decoded_q20, cell_area_median, nucleus_area_median, nucleus_to_cell_area_ratio_median
    if cells_data_by_sample or transcript_data_by_sample:

        # Add Q20+ transcripts metric if available
        if any("fraction_transcripts_decoded_q20" in data for data in xenium_module.data_by_sample.values()):
            xenium_module.genstat_headers["fraction_transcripts_decoded_q20"] = ColumnDict(
                {
                    "title": "Q20+ Transcripts",
                    "description": "Fraction of transcripts decoded with Q20+",
                    "suffix": "%",
                    "scale": "Greens",
                    "modify": lambda x: x * 100.0,
                    "max": 100.0,
                    "hidden": False,
                }
            )

        # Add cell area metrics if available
        if cells_data_by_sample:
            xenium_module.genstat_headers["cell_area_median"] = ColumnDict(
                {
                    "title": "Median Cell",
                    "description": "Median cell area",
                    "suffix": " μm²",
                    "scale": "Blues",
                    "format": "{:,.1f}",
                    "shared_key": "xenium_cell_area",
                    "hidden": False,
                }
            )

            xenium_module.genstat_headers["nucleus_area_median"] = ColumnDict(
                {
                    "title": "Median Nucleus",
                    "description": "Median nucleus area",
                    "suffix": " μm²",
                    "scale": "Oranges",
                    "format": "{:,.1f}",
                    "shared_key": "xenium_cell_area",
                    "hidden": False,
                }
            )

            xenium_module.genstat_headers["nucleus_to_cell_area_ratio_median"] = ColumnDict(
                {
                    "title": "Nucleus/Cell",
                    "description": "Median nucleus to cell area ratio",
                    "scale": "Greens",
                    "format": "{:.3f}",
                    "max": 1.0,
                    "hidden": False,
                }
            )

    # Create plots

    # 1. Segmentation Method
    segmentation_plot = xenium_segmentation_plot(xenium_module.data_by_sample)
    if segmentation_plot:
        xenium_module.add_section(
            name="Segmentation Method",
            anchor="xenium-segmentation",
            description="Distribution of cell segmentation methods used",
            helptext="""
            This stacked bar chart shows the fraction of cells segmented by each method:

            * **Boundary**: Cells segmented using boundary staining (e.g., ATP1A1/E-cadherin/CD45)
            * **Interior**: Cells segmented using interior staining (e.g., 18S RNA)
            * **Nuclear Expansion**: Cells segmented by expanding from nucleus boundaries

            **What to look for:**

            * **Boundary segmentation** typically provides the most accurate cell boundaries
            * **High nuclear expansion fraction** may indicate poor membrane staining
            * Consistent ratios across samples of the same tissue type

            **Interpretation:**

            * Over 80% boundary segmentation: Excellent membrane staining and segmentation
            * Over 50% nuclear expansion: Consider optimizing membrane staining protocols
            * Large sample-to-sample variation: Check staining consistency

            """,
            plot=segmentation_plot,
        )

    if transcript_data_by_sample:
        # 2. Transcript Quality Summary
        table_plot = xenium_transcript_quality_table(transcript_data_by_sample)
        if table_plot:
            xenium_module.add_section(
                name="Transcript Quality Summary",
                anchor="xenium-transcript-quality",
                description="Per-sample mean transcript quality statistics by gene category",
                helptext="""
                This table shows mean transcript quality statistics for each sample, with separate columns for each gene category:

                **Gene Categories:**

                * **Pre-designed**: Standard genes from Xenium panels
                * **Custom**: User-added custom targets
                * **Negative Control Probe/Codeword**: Control probes for background estimation
                * **Genomic Control Probe**: Genomic DNA controls
                * **Unassigned/Deprecated Codeword**: Other transcript types

                **Quality Score (QV) Interpretation:**

                * QV ≥20: High-quality transcripts (≥99% accuracy)
                * QV 10-20: Medium quality (90-99% accuracy)
                * QV <10: Low-quality transcripts (<90% accuracy)

                **Table Layout:**

                * **Rows**: Individual samples
                * **Columns**: Mean QV and Standard Deviation for each category
                * Values show quality statistics computed from all transcripts in that category for each sample

                **What to look for:**

                * Pre-designed genes should have high mean QV (>20) across all samples
                * Consistent quality patterns across samples indicate good data quality
                * High standard deviations may indicate heterogeneous quality within a category
                * Missing values (empty cells) indicate no transcripts found for that category in that sample
                """,
                plot=table_plot,
            )

        # 3. Distribution of Transcripts
        molecules_plot = xenium_transcripts_per_gene_plot(transcript_data_by_sample)
        if molecules_plot:
            xenium_module.add_section(
                name="Distribution of Transcripts",
                anchor="xenium-transcripts-per-gene",
                description="Distribution of transcript counts per gene",
                helptext="""
                This histogram shows the distribution of transcript counts per gene across all samples:

                **What it shows:**

                * **X-axis**: Number of transcripts per gene (log scale)
                * **Y-axis**: Number of genes with that transcript count
                * **Two categories**: Genes vs. non-genes (controls, unassigned, etc.)

                **Interpretation:**

                * **Most genes** should have moderate transcript counts (hundreds to thousands)
                * **Controls and non-genes** typically have lower counts
                * **Very high counts** may indicate highly expressed genes or technical artifacts
                * **Very low counts** may indicate poorly detected genes

                **What to look for:**

                * **Smooth distribution** for genes with a peak in the hundreds-thousands range
                * **Lower counts** for non-gene features (controls)
                * **No extreme outliers** unless biologically expected
                * **Consistent patterns** across similar tissue types

                **Quality indicators:**

                * Peak gene expression around 100-10,000 transcripts per gene is typical
                * Clear separation between gene and non-gene distributions
                * Absence of unusual spikes or gaps in the distribution
                """,
                plot=molecules_plot,
            )

    if cells_data_by_sample:
        # 4. Cell Area Distribution
        cell_area_plot = xenium_cell_area_distribution_plot(cells_data_by_sample)
        if cell_area_plot:
            xenium_module.add_section(
                name="Cell Area Distribution",
                anchor="xenium-cell-area-distribution",
                description="Distribution of cell areas across samples",
                helptext="""
                This plot shows the distribution of cell areas in the sample(s):

                * **Single sample**: Density plot with vertical lines showing mean and median cell area
                * **Multiple samples**: Violin plots showing the distribution for each sample

                **Typical cell area ranges (tissue-dependent):**

                * **Most tissues**: 50-200 μm²
                * **Large cells** (e.g., neurons): 200-500 μm²
                * **Small cells** (e.g., lymphocytes): 20-80 μm²

                **What to look for:**

                * **Consistent distributions** across samples of the same tissue type
                * **Biologically reasonable values** for your tissue
                * **Outliers**: Very large or small cells may indicate segmentation issues

                **Troubleshooting:**

                * Bimodal distributions: May indicate mixed cell types or segmentation artifacts
                * Very large cells: Over-segmentation, cell doublets, or debris
                * Very small cells: Under-segmentation, nuclear fragments
                """,
                plot=cell_area_plot,
            )

        # 5. Fraction of Transcripts in Nucleus
        nucleus_rna_plot = xenium_nucleus_rna_fraction_plot(cells_data_by_sample)
        if nucleus_rna_plot:
            xenium_module.add_section(
                name="Fraction of Transcripts in Nucleus",
                anchor="xenium-nucleus-rna-fraction",
                description="Distribution of the fraction of transcripts found in the nucleus across cells",
                helptext="""
                This plot shows the distribution of the fraction of RNA molecules located in the nucleus versus cytoplasm for each cell:

                * **Single sample**: Density plot showing the distribution of nucleus RNA fractions
                * **Multiple samples**: Box plots comparing distributions across samples

                **Biological interpretation:**

                * **Low values (0.0-0.2)**: Most RNA is cytoplasmic (expected for mature mRNAs)
                * **High values (>0.5)**: High nuclear retention (may indicate processing issues)
                * **Peak around 0.0-0.1**: Normal for most cell types with efficient RNA export

                **What to look for:**

                * **Consistent distributions** across samples of the same tissue type
                * **Biologically reasonable values** for your cell types
                * **Sample differences**: May reflect cell type composition or processing efficiency

                **Troubleshooting:**

                * Very high nuclear fractions: Check for nuclear segmentation issues
                * Bimodal distributions: May indicate different cell types or states
                * Outliers: Individual cells with unusual RNA localization patterns
                """,
                plot=nucleus_rna_plot,
            )

        # 6. Nucleus to Cell Area
        ratio_plot = xenium_nucleus_cell_area_ratio_plot(cells_data_by_sample)
        if ratio_plot:
            xenium_module.add_section(
                name="Nucleus to Cell Area",
                anchor="xenium-nucleus-cell-area-ratio",
                description="Distribution of nucleus-to-cell area ratios across cells",
                helptext="""
                This plot shows the distribution of the ratio between nucleus area and total cell area for each cell:

                * **Single sample**: Density plot showing the distribution of nucleus-to-cell area ratios
                * **Multiple samples**: Box plots comparing distributions across samples

                **Biological interpretation:**

                * **Typical range**: 0.2-0.6 for most cell types
                * **Low values (<0.2)**: Small nucleus relative to cell (may indicate active/mature cells)
                * **High values (>0.6)**: Large nucleus relative to cell (may indicate dividing or stressed cells)
                * **Peak around 0.3-0.5**: Normal for most healthy cell types

                **What to look for:**

                * **Consistent distributions** across samples of the same tissue type
                * **Biologically reasonable values** for your cell types
                * **Sample differences**: May reflect different cell states or tissue composition

                **Quality assessment:**

                * Very low ratios: May indicate over-segmented cells or debris
                * Very high ratios: May indicate under-segmented cells or nuclear fragments
                * Bimodal distributions: May indicate different cell types or segmentation artifacts

                **Troubleshooting:**

                * Unusual distributions may suggest issues with nuclear or cell segmentation parameters
                * Consider tissue-specific expected ranges when evaluating results
                """,
                plot=ratio_plot,
            )

        # 7. Distribution of Transcripts/Genes per Cell
        cell_dist_plot = xenium_cell_distributions_combined_plot(cells_data_by_sample)
        if cell_dist_plot:
            xenium_module.add_section(
                name="Distribution of Transcripts/Genes per Cell",
                anchor="xenium-cell-distributions",
                description="Distribution of transcripts and detected genes per cell",
                helptext="""
                This plot shows two key cell-level distributions with separate tabs/datasets:

                * **Tab 1: Transcripts per cell** - Shows the distribution of total transcript counts per cell
                * **Tab 2: Detected genes per cell** - Shows the distribution of unique genes detected per cell

                **Plot types:**

                * **Single sample**: Density plots showing the distribution shapes
                * **Multiple samples**: Box plots comparing distributions across samples

                **Transcripts per cell interpretation:**

                * **Typical range**: 100-5000 transcripts per cell for most tissues
                * **High transcript counts**: Metabolically active cells or large cell types
                * **Low transcript counts**: Less active cells, technical dropouts, or small cell fragments
                * **Quality thresholds**: <50 may indicate poor segmentation, >10,000 may indicate doublets

                **Detected genes per cell interpretation:**

                * **Typical range**: 50-2000 genes per cell depending on cell type and panel size
                * **High gene counts**: Metabolically active cells or cells with high expression diversity
                * **Low gene counts**: Specialized cells, inactive cells, or technical dropouts

                **What to look for:**

                * **Unimodal distributions**: Expected for homogeneous cell populations
                * **Multimodal distributions**: May indicate different cell types or technical artifacts
                * **Sample consistency**: Similar distributions expected for replicate samples
                * **Positive correlation**: Generally expect transcripts and detected genes per cell to correlate

                **Panel considerations:**

                * **Pre-designed panels**: Gene counts limited by panel design (typically 100-1000 genes)
                * **Custom panels**: Consider gene selection bias when interpreting results
                * **Detection efficiency**: Some genes may be harder to detect than others

                **Quality assessment:**

                * **Counts**: Very low (<50) or very high (>10,000) may indicate segmentation issues
                * **Shoulder distributions**: May indicate presence of different cell types

                **Troubleshooting:**

                * Unusual distributions may suggest issues with transcript detection or cell segmentation
                * Consider cell type and tissue context when evaluating expected ranges
                * Low gene detection may suggest transcript assignment issues
                """,
                plot=cell_dist_plot,
            )

    if transcript_data_by_sample:
        # 8. Field of View Quality
        fov_plot = xenium_fov_quality_plot(transcript_data_by_sample)
        if fov_plot:
            xenium_module.add_section(
                name="Field of View Quality",
                anchor="xenium-fov-quality",
                description="Field of View quality distribution across QV ranges",
                helptext="""
                This plot shows the distribution of Field of View (FoV) quality across different quality ranges:

                **What is a Field of View?**

                * Each FoV represents one microscope imaging area/tile
                * Large tissue sections are imaged as multiple overlapping FoVs
                * FoVs are systematically captured in a grid pattern across the tissue

                **Plot interpretation:**

                * **X-axis**: Quality ranges (Low to Excellent QV ranges)
                * **Y-axis**: Fields of View in each quality range
                * **Colors**: Color-coded by quality level (grey=poor, green=excellent)
                * **Bars**: Each sample shown as separate colored bars for comparison

                **Quality ranges:**

                * **Low (QV < 20)**: Poor imaging quality - investigate issues (dark grey)
                * **Poor (QV 20-25)**: Below optimal quality - may need attention (light grey)
                * **Fair (QV 25-30)**: Acceptable quality (lighter grey)
                * **Good (QV 30-35)**: Good imaging quality (light green)
                * **Excellent (QV ≥ 35)**: Optimal imaging quality (bright green)

                **What to look for:**

                * **Good distribution**: Most FoVs should be in "Good" or "Excellent" ranges
                * **Few poor FoVs**: Minimal counts in "Low" and "Poor" ranges
                * **Sample consistency**: Similar distributions across samples

                **Troubleshooting:**

                * Many low-quality FoVs: Focus/illumination issues, debris, tissue damage
                * Sample inconsistency: Processing or storage differences
                * Edge effects: FoVs at tissue edges often have lower quality
                """,
                plot=fov_plot,
            )


def xenium_segmentation_plot(data_by_sample):
    """Create stacked bar chart showing cell segmentation methods"""
    keys = {
        "segmented_cell_boundary_frac": {"name": "Boundary", "color": "#c72eba"},
        "segmented_cell_interior_frac": {"name": "Interior", "color": "#bbbf34"},
        "segmented_cell_nuc_expansion_frac": {"name": "Nuclear Expansion", "color": "#426cf5"},
    }

    plot_config = {
        "id": "xenium_segmentation",
        "title": "Xenium: Cell Segmentation Method",
        "ylab": "Fraction",
        "stacking": "normal",
        "ymax": 1.0,
        "cpswitch": False,
    }

    return bargraph.plot(data_by_sample, keys, plot_config)


def parse_transcripts_parquet(f) -> Optional[Dict]:
    """Parse Xenium transcripts.parquet file with optimized lazy dataframe processing

    Only computes aggregated statistics needed for reporting, avoiding per-transcript dictionaries.

    Args:
        f: File info dict
    """
    file_path = Path(f["root"]) / f["fn"]

    # Use lazy loading to avoid reading entire file into memory
    df_lazy = pl.scan_parquet(file_path)

    # Check if required columns exist by scanning schema (avoid performance warning)
    schema = df_lazy.collect_schema()
    required_cols = ["qv", "feature_name"]
    if not all(col in schema for col in required_cols):
        log.warning(f"Missing required columns in {f['fn']}: {required_cols}")
        return None

    # Get total row count efficiently without loading full data
    total_transcripts = df_lazy.select(pl.len()).collect().item()

    # Compute category statistics directly in lazy dataframe for optimal performance
    # This replaces per-transcript dictionaries with aggregated category stats
    category_stats = (
        df_lazy.with_columns(
            pl.col("feature_name")
            .map_elements(lambda x: categorize_feature(str(x))[0], return_dtype=pl.Utf8)
            .alias("category")
        )
        .group_by("category")
        .agg(
            [
                pl.col("qv").mean().alias("mean_quality"),
                pl.col("qv").std().alias("std_quality"),
                pl.col("qv").count().alias("transcript_count"),
                pl.col("feature_name").n_unique().alias("feature_count"),
            ]
        )
        .collect()
    )

    # Create optimized result structure - only store aggregated category statistics
    category_summary = {}
    for row in category_stats.iter_rows(named=True):
        category = str(row["category"])
        category_summary[category] = {
            "mean_quality": row["mean_quality"],
            "std_quality": row["std_quality"] or 0.0,  # Handle null std for single values
            "transcript_count": row["transcript_count"],
            "feature_count": row["feature_count"],
        }

    result = {
        "category_summary": category_summary,
        "total_transcripts": total_transcripts,
    }

    # Add feature-level transcript counts for scatter plot (single sample case)
    # This is needed for the transcript quality scatter plot
    feature_stats = (
        df_lazy.group_by("feature_name")
        .agg(
            [
                pl.col("qv").mean().alias("mean_quality"),
                pl.col("qv").count().alias("count"),
            ]
        )
        .collect()
    )

    # Create transcript_counts dictionary for scatter plot
    transcript_counts = {}
    for row in feature_stats.iter_rows(named=True):
        feature_name = str(row["feature_name"])
        transcript_counts[feature_name] = {
            "count": row["count"],
            "mean_quality": row["mean_quality"],
        }

    result["transcript_counts"] = transcript_counts

    # Add transcripts per gene analysis if is_gene column is present
    if "is_gene" in schema:
        transcript_stats = (
            df_lazy.group_by("feature_name")
            .agg([pl.len().alias("transcript_count"), pl.col("is_gene").first().alias("is_gene")])
            .collect()
        )

        if not transcript_stats.is_empty():
            molecules_per_gene = {}
            for row in transcript_stats.iter_rows(named=True):
                feature_name = str(row["feature_name"])
                molecules_per_gene[feature_name] = {
                    "count": row["transcript_count"],  # This is transcript count per gene
                    "is_gene": row["is_gene"],
                }
            result["molecules_per_gene"] = molecules_per_gene

            # Calculate noise threshold directly from transcript_stats DataFrame
            result["noise_threshold"] = calculate_noise_threshold_from_df(transcript_stats)

    # Add FoV quality analysis if fov_name column is present
    if "fov_name" in schema:
        fov_stats = (
            df_lazy.group_by("fov_name")
            .agg(
                [
                    pl.col("qv").mean().alias("mean_qv"),
                    pl.col("qv").median().alias("median_qv"),
                    pl.col("qv").std().alias("std_qv"),
                    pl.len().alias("transcript_count"),
                ]
            )
            .collect()
        )

        fov_quality_stats = {}
        fov_medians = []
        for row in fov_stats.iter_rows(named=True):
            fov_name = str(row["fov_name"])
            median_qv = row["median_qv"]
            fov_quality_stats[fov_name] = {
                "mean_quality": row["mean_qv"],
                "median_quality": median_qv,
                "std_quality": row["std_qv"] or 0.0,
                "transcript_count": row["transcript_count"],
            }
            if median_qv is not None:
                fov_medians.append(median_qv)

        result["fov_quality_stats"] = fov_quality_stats
        result["fov_median_qualities"] = fov_medians  # For heatmap generation

    return result


def parse_cells_parquet(f) -> Optional[Dict]:
    """Parse Xenium cells.parquet file to extract cell-level metrics"""
    file_path = Path(f["root"]) / f["fn"]

    # Use lazy reading to avoid loading entire file into memory
    log.debug(f"Processing xenium cells parquet file: {file_path}")
    # Start with lazy frame to check schema without loading data
    lazy_df = pl.scan_parquet(file_path, parallel="none")  # parallel execution causing panics

    # Check for required columns using schema
    schema = lazy_df.collect_schema()
    required_cols = ["cell_area", "nucleus_area", "total_counts", "transcript_counts"]
    missing_cols = [col for col in required_cols if col not in schema]
    if missing_cols:
        log.warning(f"Missing columns in {f['fn']}: {missing_cols}")
        return None

    # Get row count efficiently without loading data
    total_cells = lazy_df.select(pl.len()).collect().item()
    cell_stats = {"total_cells": total_cells}

    # Cell area distribution stats using lazy operations
    cell_area_stats = (
        lazy_df.filter(pl.col("cell_area").is_not_null())
        .select(
            [
                pl.col("cell_area").mean().alias("mean"),
                pl.col("cell_area").median().alias("median"),
                pl.col("cell_area").std().alias("std"),
                pl.col("cell_area").min().alias("min"),
                pl.col("cell_area").max().alias("max"),
                pl.col("cell_area").quantile(0.25).alias("q1"),
                pl.col("cell_area").quantile(0.75).alias("q3"),
                pl.col("cell_area").count().alias("count"),
            ]
        )
        .collect()
    )

    if cell_area_stats["count"].item() > 0:
        cell_stats.update(
            {
                "cell_area_mean": cell_area_stats["mean"].item(),
                "cell_area_median": cell_area_stats["median"].item(),
                "cell_area_std": cell_area_stats["std"].item(),
                "cell_area_min": cell_area_stats["min"].item(),
                "cell_area_max": cell_area_stats["max"].item(),
                "cell_area_q1": cell_area_stats["q1"].item(),
                "cell_area_q3": cell_area_stats["q3"].item(),
            }
        )

        # Store box plot statistics instead of raw values
        cell_stats["cell_area_box_stats"] = {
            "min": cell_area_stats["min"].item(),
            "q1": cell_area_stats["q1"].item(),
            "median": cell_area_stats["median"].item(),
            "q3": cell_area_stats["q3"].item(),
            "max": cell_area_stats["max"].item(),
            "mean": cell_area_stats["mean"].item(),
            "count": cell_area_stats["count"].item(),
        }

    # Nucleus area distribution stats using lazy operations
    nucleus_area_stats = (
        lazy_df.filter(pl.col("nucleus_area").is_not_null())
        .select(
            [
                pl.col("nucleus_area").mean().alias("mean"),
                pl.col("nucleus_area").median().alias("median"),
                pl.col("nucleus_area").std().alias("std"),
                pl.col("nucleus_area").count().alias("count"),
            ]
        )
        .collect()
    )

    if nucleus_area_stats["count"].item() > 0:
        cell_stats.update(
            {
                "nucleus_area_mean": nucleus_area_stats["mean"].item(),
                "nucleus_area_median": nucleus_area_stats["median"].item(),
                "nucleus_area_std": nucleus_area_stats["std"].item(),
            }
        )

        # Nucleus to cell area ratio (only for non-null values)
        ratio_stats = (
            lazy_df.filter(
                (pl.col("cell_area").is_not_null())
                & (pl.col("nucleus_area").is_not_null())
                & (pl.col("cell_area") > 0)
            )
            .with_columns((pl.col("nucleus_area") / pl.col("cell_area")).alias("ratio"))
            .select(
                [
                    pl.col("ratio").mean().alias("mean"),
                    pl.col("ratio").median().alias("median"),
                    pl.col("ratio").count().alias("count"),
                ]
            )
            .collect()
        )

        if ratio_stats["count"].item() > 0:
            cell_stats.update(
                {
                    "nucleus_to_cell_area_ratio_mean": ratio_stats["mean"].item(),
                    "nucleus_to_cell_area_ratio_median": ratio_stats["median"].item(),
                }
            )

            # Calculate ratio distribution statistics for box plots
            ratio_dist_stats = (
                lazy_df.filter(
                    (pl.col("cell_area").is_not_null())
                    & (pl.col("nucleus_area").is_not_null())
                    & (pl.col("cell_area") > 0)
                )
                .with_columns((pl.col("nucleus_area") / pl.col("cell_area")).alias("ratio"))
                .select(
                    [
                        pl.col("ratio").min().alias("min"),
                        pl.col("ratio").quantile(0.25).alias("q1"),
                        pl.col("ratio").median().alias("median"),
                        pl.col("ratio").quantile(0.75).alias("q3"),
                        pl.col("ratio").max().alias("max"),
                        pl.col("ratio").mean().alias("mean"),
                        pl.col("ratio").count().alias("count"),
                    ]
                )
                .collect()
            )

            if ratio_dist_stats["count"].item() > 0:
                cell_stats["nucleus_to_cell_area_ratio_box_stats"] = {
                    "min": ratio_dist_stats["min"].item(),
                    "q1": ratio_dist_stats["q1"].item(),
                    "median": ratio_dist_stats["median"].item(),
                    "q3": ratio_dist_stats["q3"].item(),
                    "max": ratio_dist_stats["max"].item(),
                    "mean": ratio_dist_stats["mean"].item(),
                    "count": ratio_dist_stats["count"].item(),
                }

    # Store total transcript counts per cell (total_counts) for distribution plots
    total_count_check = (
        lazy_df.filter(pl.col("total_counts").is_not_null())
        .select(pl.col("total_counts").count().alias("count"))
        .collect()
    )

    if total_count_check["count"].item() > 0:
        # Calculate total counts distribution statistics for box plots
        total_counts_stats = (
            lazy_df.filter(pl.col("total_counts").is_not_null())
            .select(
                [
                    pl.col("total_counts").min().alias("min"),
                    pl.col("total_counts").quantile(0.25).alias("q1"),
                    pl.col("total_counts").median().alias("median"),
                    pl.col("total_counts").quantile(0.75).alias("q3"),
                    pl.col("total_counts").max().alias("max"),
                    pl.col("total_counts").mean().alias("mean"),
                    pl.col("total_counts").count().alias("count"),
                ]
            )
            .collect()
        )
        cell_stats["total_counts_box_stats"] = {
            "min": total_counts_stats["min"].item(),
            "q1": total_counts_stats["q1"].item(),
            "median": total_counts_stats["median"].item(),
            "q3": total_counts_stats["q3"].item(),
            "max": total_counts_stats["max"].item(),
            "mean": total_counts_stats["mean"].item(),
            "count": total_counts_stats["count"].item(),
        }

    # Store detected genes per cell (transcript_counts) for distribution plots
    # NOTE: This will be overridden by H5-based calculation if cell_feature_matrix.h5 is available
    detected_count_check = (
        lazy_df.filter(pl.col("transcript_counts").is_not_null())
        .select(pl.col("transcript_counts").count().alias("count"))
        .collect()
    )

    if detected_count_check["count"].item() > 0:
        # Calculate detected genes per cell distribution statistics for box plots
        gene_counts_stats = (
            lazy_df.filter(pl.col("transcript_counts").is_not_null())
            .select(
                [
                    pl.col("transcript_counts").min().alias("min"),
                    pl.col("transcript_counts").quantile(0.25).alias("q1"),
                    pl.col("transcript_counts").median().alias("median"),
                    pl.col("transcript_counts").quantile(0.75).alias("q3"),
                    pl.col("transcript_counts").max().alias("max"),
                    pl.col("transcript_counts").mean().alias("mean"),
                    pl.col("transcript_counts").count().alias("count"),
                ]
            )
            .collect()
        )
        cell_stats["gene_transcript_counts_box_stats"] = {
            "min": gene_counts_stats["min"].item(),
            "q1": gene_counts_stats["q1"].item(),
            "median": gene_counts_stats["median"].item(),
            "q3": gene_counts_stats["q3"].item(),
            "max": gene_counts_stats["max"].item(),
            "mean": gene_counts_stats["mean"].item(),
            "count": gene_counts_stats["count"].item(),
        }

    # Add nucleus RNA fraction if nucleus_count is available
    if "nucleus_count" in schema:
        nucleus_fraction_stats = (
            lazy_df.filter(pl.col("total_counts") >= 10)
            .with_columns((pl.col("nucleus_count") / pl.col("total_counts")).alias("fraction"))
            .select(
                [
                    pl.col("fraction").mean().alias("mean"),
                    pl.col("fraction").median().alias("median"),
                    pl.col("fraction").count().alias("count"),
                ]
            )
            .collect()
        )

        if nucleus_fraction_stats["count"].item() > 0:
            cell_stats.update(
                {
                    "nucleus_rna_fraction_mean": nucleus_fraction_stats["mean"].item(),
                    "nucleus_rna_fraction_median": nucleus_fraction_stats["median"].item(),
                }
            )

            # Calculate nucleus RNA fraction distribution statistics for box plots
            nucleus_fraction_dist_stats = (
                lazy_df.filter(pl.col("total_counts") > 0)
                .with_columns((pl.col("nucleus_count") / pl.col("total_counts")).alias("fraction"))
                .select(
                    [
                        pl.col("fraction").min().alias("min"),
                        pl.col("fraction").quantile(0.25).alias("q1"),
                        pl.col("fraction").median().alias("median"),
                        pl.col("fraction").quantile(0.75).alias("q3"),
                        pl.col("fraction").max().alias("max"),
                        pl.col("fraction").mean().alias("mean"),
                        pl.col("fraction").count().alias("count"),
                    ]
                )
                .collect()
            )
            cell_stats["nucleus_rna_fraction_box_stats"] = {
                "min": nucleus_fraction_dist_stats["min"].item(),
                "q1": nucleus_fraction_dist_stats["q1"].item(),
                "median": nucleus_fraction_dist_stats["median"].item(),
                "q3": nucleus_fraction_dist_stats["q3"].item(),
                "max": nucleus_fraction_dist_stats["max"].item(),
                "mean": nucleus_fraction_dist_stats["mean"].item(),
                "count": nucleus_fraction_dist_stats["count"].item(),
            }

    return cell_stats


def parse_cell_feature_matrix_h5(f):
    """Parse cell_feature_matrix.h5 file to calculate detected genes per cell"""
    try:
        # Construct full file path
        file_path = Path(f["root"]) / f["fn"]

        # Read H5 file using scanpy
        adata = sc.read_10x_h5(str(file_path))

        # Calculate detected genes per cell (number of non-zero genes per cell)
        # This matches the notebook's approach: (ad.X != 0).sum(axis=1).A1
        n_genes_per_cell = (adata.X != 0).sum(axis=1).A1

        result = {}

        # Calculate statistics for detected genes per cell (similar to transcript_counts processing)
        if len(n_genes_per_cell) > 0:
            detected_genes_stats = {
                "min": float(np.min(n_genes_per_cell)),
                "q1": float(np.percentile(n_genes_per_cell, 25)),
                "median": float(np.median(n_genes_per_cell)),
                "q3": float(np.percentile(n_genes_per_cell, 75)),
                "max": float(np.max(n_genes_per_cell)),
                "mean": float(np.mean(n_genes_per_cell)),
                "count": len(n_genes_per_cell),
            }

            # Store as gene_transcript_counts_box_stats to replace the current implementation
            result["detected_genes_stats"] = detected_genes_stats

            # Also store raw values if needed for single-sample density plots
            result["detected_genes_values"] = n_genes_per_cell.tolist()

            log.debug(f"Processed {file_path}: {len(n_genes_per_cell)} cells, {adata.n_vars} genes")
            log.debug(
                f"Detected genes per cell - mean: {detected_genes_stats['mean']:.1f}, median: {detected_genes_stats['median']:.1f}"
            )

        return result

    except Exception as e:
        log.warning(f"Failed to process {f.get('fn', 'cell_feature_matrix.h5')}: {str(e)}")
        return None


def xenium_transcript_quality_scatter_plot(transcript_data_by_sample):
    """Create scatter plot - handles both single and multiple samples"""
    # Prepare scatter data - create individual points for each gene from all samples
    plot_data: Dict[str, Any] = {}

    for sample_name, sample_data in transcript_data_by_sample.items():
        if "transcript_counts" not in sample_data:
            continue

        for feature, counts_data in sample_data["transcript_counts"].items():
            category, feature_id = categorize_feature(feature)

            if category not in plot_data:
                plot_data[category] = []

            # Each point is a separate data point
            # For multiple samples, include sample name in the hover text
            if len(transcript_data_by_sample) > 1:
                point_name = f"{feature_id} ({sample_name})"
            else:
                point_name = feature_id

            plot_data[category].append(
                {
                    "x": counts_data["count"],
                    "y": counts_data["mean_quality"],
                    "name": point_name,  # Use gene name (+ sample) for hover text
                    "group": category,
                }
            )

    # Filter out empty categories and add colors to each point
    final_plot_data = {}
    for category, points in plot_data.items():
        if points:  # Only include categories with data
            # Add color to each point in the category
            for point in points:
                point["color"] = GENE_CATS[category]["color"]
            final_plot_data[category] = points

    # Adjust title based on number of samples
    if len(transcript_data_by_sample) == 1:
        title = "Xenium: Gene-Specific Transcript Quality"
    else:
        title = f"Xenium: Gene-Specific Transcript Quality ({len(transcript_data_by_sample)} samples)"

    # Define desired category order for legend
    category_order = [
        "Pre-designed",
        "Custom",
        "Genomic Control Probe",
        "Negative Control Probe",
        "Negative Control Codeword",
        "Unassigned Codeword",
        "Deprecated Codeword",
    ]

    config = {
        "id": "xenium_transcript_quality_combined",
        "title": title,
        "xlab": "Total transcripts per gene",
        "ylab": "Mean calibrated quality of gene transcripts",
        "marker_size": 4,
        "marker_line_width": 0,
        "opacity": 0.75,
        "series_label": "transcripts",
        "xlog": True,
        "showlegend": True,
        "groups": category_order,
        "flat_if_very_large": False,
    }

    return scatter.plot(final_plot_data, config)


def xenium_transcript_quality_table(transcript_data_by_sample):
    """Create per-sample table showing mean quality for each category (samples as rows, categories as columns)"""
    if not transcript_data_by_sample:
        return None

    # Collect all categories across samples to create consistent columns
    all_categories = set()
    for sample_data in transcript_data_by_sample.values():
        if "category_summary" in sample_data:
            all_categories.update(sample_data["category_summary"].keys())

    if not all_categories:
        return None

    # Sort categories for consistent ordering
    sorted_categories = sorted(
        all_categories,
        key=lambda x: (
            0
            if x == "Pre-designed"
            else 1
            if x == "Custom"
            else 2
            if x == "Genomic Control Probe"
            else 3
            if x == "Negative Control Probe"
            else 4
            if x == "Negative Control Codeword"
            else 5
            if x == "Unassigned Codeword"
            else 6
            if x == "Deprecated Codeword"
            else 7
        ),
    )

    # Create table data: samples as rows, categories as columns
    table_data = {}
    for sample_name, sample_data in transcript_data_by_sample.items():
        if "category_summary" not in sample_data:
            continue

        table_data[sample_name] = {}

        # Add mean quality for each category
        for category in sorted_categories:
            if category in sample_data["category_summary"]:
                mean_quality = sample_data["category_summary"][category]["mean_quality"]
                table_data[sample_name][f"{category} Mean QV"] = mean_quality
            else:
                table_data[sample_name][f"{category} Mean QV"] = None

        # Add standard deviation for each category
        for category in sorted_categories:
            if category in sample_data["category_summary"]:
                std_quality = sample_data["category_summary"][category]["std_quality"]
                table_data[sample_name][f"{category} Std Dev"] = std_quality
            else:
                table_data[sample_name][f"{category} Std Dev"] = None

    if not table_data:
        return None

    # Create table headers for each category (both mean and std dev)
    headers: Dict[str, ColumnDict] = {}

    # Create consistent abbreviations for column titles
    category_abbreviations = {
        "Pre-designed": "Pre-designed",
        "Custom": "Custom",
        "Genomic Сontrol Probe": "Genomic Ctrl",
        "Negative Control Probe": "Negative Ctrl",
        "Negative Control Codeword": "Neg Codeword",
        "Unassigned Codeword": "Unassigned",
        "Deprecated Codeword": "Deprecated",
    }

    for category in sorted_categories:
        # Get abbreviated title for consistent column width
        abbrev_title = category_abbreviations.get(category, category)

        # Mean quality column
        headers[f"{category} Mean QV"] = {
            "title": f"{abbrev_title}",
            "description": f"Mean calibrated quality score (QV) for {category}",
            "scale": "Blues",
            "format": "{:.2f}",
            "suffix": "",
            "shared_key": "xenium_transcript_quality",
            "min": 0,
            "max": 40,
        }

        # Standard deviation column
        headers[f"{category} Std Dev"] = {
            "title": f"{abbrev_title} StdDev",
            "description": f"Standard deviation of quality scores for {category}",
            "scale": "Oranges",
            "format": "{:.2f}",
            "suffix": "",
            "shared_key": "xenium_transcript_quality",
            "hidden": True,
        }

    return table.plot(
        table_data,
        headers,
        pconfig=TableConfig(
            id="xenium_transcript_quality_per_sample_table",
            title="Xenium: Mean Transcript Quality by Sample and Category",
        ),
    )


def xenium_fov_quality_plot(transcript_data_by_sample):
    """Create bar plot showing FoV count distribution across QV ranges"""
    # Collect median quality per FoV per sample
    fov_median_by_sample = {}

    for s_name, data in transcript_data_by_sample.items():
        data = transcript_data_by_sample[s_name]
        if "fov_quality_stats" in data:
            fov_median_by_sample[s_name] = {}
            fov_stats = data["fov_quality_stats"]
            for fov_name, stats in fov_stats.items():
                median_quality = stats["median_quality"]
                if median_quality is not None:
                    fov_median_by_sample[s_name][fov_name] = median_quality

    if not fov_median_by_sample:
        return None

    # Define QV ranges (ordered high to low for display)
    qv_ranges = [
        ("Excellent (QV ≥ 35)", 35, float("inf")),
        ("Good (QV 30-35)", 30, 35),
        ("Fair (QV 25-30)", 25, 30),
        ("Poor (QV 20-25)", 20, 25),
        ("Low (QV < 20)", 0, 20),
    ]

    # Create bar plot data - count FoVs in each QV range per sample
    bar_data = {}
    for sample_name, fov_qualities in fov_median_by_sample.items():
        bar_data[sample_name] = {}

        # Initialize counts for each range
        for range_name, _, _ in qv_ranges:
            bar_data[sample_name][range_name] = 0

        # Count FoVs in each range
        for fov_name, quality in fov_qualities.items():
            for range_name, min_qv, max_qv in qv_ranges:
                if min_qv <= quality < max_qv:
                    bar_data[sample_name][range_name] += 1
                    break

    config = {
        "id": "xenium_fov_quality_ranges",
        "title": "Xenium: Field of View Quality Distribution",
        "xlab": "Quality Range",
        "ylab": "Fields of View",
        "cpswitch_c_active": False,
        "use_legend": True,
    }

    # Define categories with colors (grey-to-green gradient, ordered high to low)
    cats = {
        "Excellent (QV ≥ 35)": {
            "name": "Excellent (QV ≥ 35)",
            "color": "rgba(0, 216, 0, 0.71)",  # Bright green for excellent quality
        },
        "Good (QV 30-35)": {
            "name": "Good (QV 30-35)",
            "color": "rgba(98, 196, 98, 0.52)",  # Light green for good quality
        },
        "Fair (QV 25-30)": {
            "name": "Fair (QV 25-30)",
            "color": "rgba(177, 111, 119, 0.55)",  # Light pink for fair quality
        },
        "Poor (QV 20-25)": {
            "name": "Poor (QV 20-25)",
            "color": "rgba(176, 37, 46, 0.64)",  # Medium pink-red for poor quality
        },
        "Low (QV < 20)": {
            "name": "Low (QV < 20)",
            "color": "rgba(255, 0, 51, 0.67)",  # Dark red for low quality
        },
    }

    return bargraph.plot(bar_data, cats, config)


def xenium_transcripts_per_gene_plot(transcript_data_by_sample):
    """Create histogram plot showing distribution of transcripts per gene with separate lines per sample"""
    # Check if any sample has molecules per gene data
    samples_with_molecules = []
    for s_name, data in transcript_data_by_sample.items():
        if "molecules_per_gene" in data:
            samples_with_molecules.append(s_name)

    if not samples_with_molecules:
        return None

    # Determine if single or multi-sample plot
    num_samples = len(samples_with_molecules)
    if num_samples == 1:
        # Single sample: calculate noise threshold for this sample only
        s_name = samples_with_molecules[0]
        sample_data = transcript_data_by_sample[s_name]
        molecules_data = sample_data["molecules_per_gene"]

        # Use pre-calculated noise threshold if available
        n_mols_threshold = sample_data.get("noise_threshold")
    else:
        # Multi-sample: use pre-calculated noise thresholds
        sample_thresholds = {}
        for s_name in samples_with_molecules:
            sample_data = transcript_data_by_sample[s_name]

            # Use pre-calculated noise threshold if available
            threshold = sample_data.get("noise_threshold")
            sample_thresholds[s_name] = threshold

        n_mols_threshold = None  # Keep for single-sample compatibility

    # Determine global bins based on all samples' data
    all_gene_counts = []
    all_non_gene_counts = []

    for s_name in samples_with_molecules:
        data = transcript_data_by_sample[s_name]
        molecules_data = data["molecules_per_gene"]

        for _, gene_info in molecules_data.items():
            count = gene_info["count"]
            if count > 0:
                if gene_info["is_gene"]:
                    all_gene_counts.append(count)
                else:
                    all_non_gene_counts.append(count)

    # Create consistent bins for all samples
    all_counts = all_gene_counts + all_non_gene_counts
    if not all_counts:
        return None

    min_count = max(1, min(all_counts))
    max_count = max(all_counts)
    bins = np.logspace(np.log10(min_count), np.log10(max_count), 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Choose between single and multi-sample plots
    if num_samples == 1:
        # Single sample with noise threshold
        s_name = samples_with_molecules[0]
        sample_data = transcript_data_by_sample[s_name]
        # Create single-item threshold dict for consistency
        single_sample_thresholds = {s_name: n_mols_threshold}
        return _create_single_sample_molecules_plot(
            sample_data, bins, bin_centers, single_sample_thresholds, s_name
        )
    else:
        # Multi-sample with per-sample thresholds
        return _create_multi_sample_molecules_plot(
            transcript_data_by_sample, samples_with_molecules, bins, bin_centers, sample_thresholds
        )


def _create_single_sample_molecules_plot(sample_data, bins, bin_centers, sample_thresholds, sample_name):
    """Create single plot with both Gene and Non-gene lines for single sample"""
    molecules_data = sample_data["molecules_per_gene"]

    # Separate counts by gene type
    gene_counts = []
    non_gene_counts = []

    for _, gene_info in molecules_data.items():
        count = gene_info["count"]
        if count > 0:
            if gene_info["is_gene"]:
                gene_counts.append(count)
            else:
                non_gene_counts.append(count)

    # Create plot data with both lines
    plot_data = {}
    all_histograms = []

    if gene_counts:
        gene_hist, _ = np.histogram(gene_counts, bins=bins)
        all_histograms.append(gene_hist)
        gene_line_data = {}
        for i, count in enumerate(gene_hist):
            gene_line_data[float(bin_centers[i])] = int(count)
        plot_data["Genes"] = gene_line_data

    if non_gene_counts:
        non_gene_hist, _ = np.histogram(non_gene_counts, bins=bins)
        all_histograms.append(non_gene_hist)
        non_gene_line_data = {}
        for i, count in enumerate(non_gene_hist):
            non_gene_line_data[float(bin_centers[i])] = int(count)
        plot_data["Non-genes"] = non_gene_line_data

    if not plot_data:
        return None

    # Trim long tail: find cutoff where all values above X are below 1% of max
    if all_histograms:
        # Get maximum value across all histograms
        max_value = max(np.max(hist) for hist in all_histograms)
        threshold = max_value * 0.01  # 1% of max

        # Find the last bin where any histogram has values above threshold
        last_significant_bin = len(bin_centers) - 1
        for i in range(len(bin_centers) - 1, -1, -1):
            if any(hist[i] >= threshold for hist in all_histograms):
                last_significant_bin = i
                break

        # Trim the data to only include up to the last significant bin
        if last_significant_bin < len(bin_centers) - 1:
            trimmed_plot_data = {}
            for dataset_name, data in plot_data.items():
                trimmed_data = {}
                for i, (x_val, y_val) in enumerate(data.items()):
                    if i <= last_significant_bin:
                        trimmed_data[x_val] = y_val
                trimmed_plot_data[dataset_name] = trimmed_data
            plot_data = trimmed_plot_data

    config: Dict[str, Any] = {
        "id": "xenium_transcripts_per_gene",
        "title": "Xenium: Distribution of Transcripts per Gene",
        "xlab": "Number of transcripts per gene",
        "ylab": "Number of features",
        "xlog": True,
        "series_label": False,
    }

    # Use same color for genes and controls from same sample (distinguished by line style)
    scale = mqc_colour.mqc_colour_scale("plot_defaults")
    sample_color = scale.get_colour(0, lighten=1)  # Use first color for single sample

    n_mols_threshold = sample_thresholds.get(sample_name) if sample_thresholds else None
    threshold_text = f" (noise threshold: {n_mols_threshold:.0f})" if n_mols_threshold is not None else ""

    colors = {
        "Genes": sample_color,
    }
    config["colors"] = colors

    # Use dash_styles and hovertemplates for series styling
    if "Non-genes" in plot_data:
        colors["Non-genes"] = sample_color  # Same color as genes
        config["dash_styles"] = {
            "Genes": "solid",
            "Non-genes": "dash",  # Dashed line for controls
        }
        config["hovertemplates"] = {
            "Genes": f"<b>%{{text}}</b><br>%{{x}}: %{{y}}{threshold_text}<extra></extra>",
            "Non-genes": f"<b>%{{text}}</b><br>%{{x}}: %{{y}}{threshold_text}<extra></extra>",
        }
        config["legend_groups"] = {"Genes": sample_name, "Non-genes": sample_name}
    else:
        config["hovertemplates"] = {"Genes": f"<b>%{{text}}</b><br>%{{x}}: %{{y}}{threshold_text}<extra></extra>"}
        config["legend_groups"] = {"Genes": sample_name}

    # Add vertical line for noise threshold if calculated
    if n_mols_threshold is not None and n_mols_threshold > 0:
        config["x_lines"] = [
            {
                "value": n_mols_threshold,
                "color": "grey",
                "dash": "dash",
                "width": 1,
                "label": f"Noise threshold ({n_mols_threshold:.0f})",
            }
        ]

    return linegraph.plot(plot_data, config)


def _create_multi_sample_molecules_plot(
    transcript_data_by_sample, samples_with_molecules, bins, bin_centers, sample_thresholds
):
    """Create single plot with all samples shown as separate lines, color-coded by gene type"""
    plot_data = {}
    all_histograms = []

    # Process each sample and separate by gene type
    for s_name in samples_with_molecules:
        data = transcript_data_by_sample[s_name]
        molecules_data = data["molecules_per_gene"]

        # Separate this sample's counts by gene type
        sample_gene_counts = []
        sample_non_gene_counts = []

        for _, gene_info in molecules_data.items():
            count = gene_info["count"]
            if count > 0:
                if gene_info["is_gene"]:
                    sample_gene_counts.append(count)
                else:
                    sample_non_gene_counts.append(count)

        # Create histograms for genes (blue lines)
        if sample_gene_counts:
            gene_hist, _ = np.histogram(sample_gene_counts, bins=bins)
            all_histograms.append(gene_hist)
            gene_line_data = {}
            for i, count in enumerate(gene_hist):
                gene_line_data[float(bin_centers[i])] = int(count)
            plot_data[f"{s_name} (genes)"] = gene_line_data

        # Create histograms for non-genes (black lines)
        if sample_non_gene_counts:
            non_gene_hist, _ = np.histogram(sample_non_gene_counts, bins=bins)
            all_histograms.append(non_gene_hist)
            non_gene_line_data = {}
            for i, count in enumerate(non_gene_hist):
                non_gene_line_data[float(bin_centers[i])] = int(count)
            plot_data[f"{s_name} (non-genes)"] = non_gene_line_data

    if not plot_data:
        return None

    # Trim long tail: find cutoff where all values above X are below 1% of max
    if all_histograms:
        # Get maximum value across all histograms
        max_value = max(np.max(hist) for hist in all_histograms)
        threshold = max_value * 0.01  # 1% of max

        # Find the last bin where any histogram has values above threshold
        last_significant_bin = len(bin_centers) - 1
        for i in range(len(bin_centers) - 1, -1, -1):
            if any(hist[i] >= threshold for hist in all_histograms):
                last_significant_bin = i
                break

        # Trim the data to only include up to the last significant bin
        if last_significant_bin < len(bin_centers) - 1:
            trimmed_plot_data = {}
            for dataset_name, data in plot_data.items():
                trimmed_data = {}
                for i, (x_val, y_val) in enumerate(data.items()):
                    if i <= last_significant_bin:
                        trimmed_data[x_val] = y_val
                trimmed_plot_data[dataset_name] = trimmed_data
            plot_data = trimmed_plot_data

    config: Dict[str, Any] = {
        "id": "xenium_transcripts_per_gene",
        "title": "Xenium: Distribution of Transcripts per Gene",
        "xlab": "Number of transcripts per gene",
        "ylab": "Number of features",
        "series_label": False,
        "xlog": True,
        "x_decimals": 0,
    }

    # Use per-sample coloring with mqc_colour plot_defaults scheme
    scale = mqc_colour.mqc_colour_scale("plot_defaults")

    # Group paired lines by sample name and assign colors
    sample_names = set()
    for dataset_name in plot_data.keys():
        if "(genes)" in dataset_name:
            sample_name = dataset_name.replace(" (genes)", "")
            sample_names.add(sample_name)
        elif "(non-genes)" in dataset_name:
            sample_name = dataset_name.replace(" (non-genes)", "")
            sample_names.add(sample_name)

    # Create color mapping for each sample
    sample_colors = {}
    for idx, sample_name in enumerate(sorted(sample_names)):
        sample_colors[sample_name] = scale.get_colour(idx, lighten=1)

    # Use the new parameters to style series instead of extra_series
    colors = {}
    dash_styles = {}
    hovertemplates = {}
    legend_groups = {}

    # Set up styling for all series using the new parameters
    for dataset_name in plot_data.keys():
        if "(genes)" in dataset_name:
            sample_name = dataset_name.replace(" (genes)", "")
            threshold = sample_thresholds.get(sample_name)
            threshold_text = f" (noise threshold: {threshold:.0f})" if threshold is not None else ""

            colors[dataset_name] = sample_colors[sample_name]
            dash_styles[dataset_name] = "solid"  # Solid lines for genes
            hovertemplates[dataset_name] = f"<b>%{{text}}</b><br>%{{x}}: %{{y}}{threshold_text}<extra></extra>"
            legend_groups[dataset_name] = sample_name

        elif "(non-genes)" in dataset_name:
            sample_name = dataset_name.replace(" (non-genes)", "")
            threshold = sample_thresholds.get(sample_name)
            threshold_text = f" (noise threshold: {threshold:.0f})" if threshold is not None else ""

            colors[dataset_name] = sample_colors[sample_name]
            dash_styles[dataset_name] = "dash"  # Dashed lines for controls
            hovertemplates[dataset_name] = f"<b>%{{text}}</b><br>%{{x}}: %{{y}}{threshold_text}<extra></extra>"
            legend_groups[dataset_name] = sample_name

    config["colors"] = colors
    config["dash_styles"] = dash_styles
    config["hovertemplates"] = hovertemplates
    config["legend_groups"] = legend_groups

    return linegraph.plot(plot_data, config)


def calculate_noise_threshold_from_df(transcript_stats_df, quantile=0.99):
    """
    Calculate noise threshold directly from transcript_stats DataFrame.

    Args:
        transcript_stats_df: Polars DataFrame with columns ['feature_name', 'transcript_count', 'is_gene']
        quantile: Quantile for threshold calculation (default 0.99)

    Returns:
        Float threshold value or None if insufficient data
    """
    # Filter for negative control features using polars
    neg_controls = transcript_stats_df.filter(
        (~pl.col("is_gene")) & pl.col("feature_name").str.starts_with("NegControl")
    )

    # Get counts > 0 for negative controls
    neg_control_counts = neg_controls.filter(pl.col("transcript_count") > 0)["transcript_count"].to_list()

    if len(neg_control_counts) < 3:  # Need at least 3 data points for meaningful statistics
        return None

    # Calculate upper bound using quantile

    # Calculate threshold using log-space statistics (similar to notebook)
    log_counts = np.log10(neg_control_counts)

    # Use median absolute deviation as robust estimate of standard deviation
    median_log = np.median(log_counts)
    mad = np.median(np.abs(log_counts - median_log))
    # Convert MAD to standard deviation equivalent (normal distribution scaling factor)
    std_log = mad * 1.4826

    z_score = stats.norm.ppf(quantile)
    threshold_log = median_log + z_score * std_log

    threshold = 10**threshold_log
    return threshold


def xenium_cell_distributions_combined_plot(cells_data_by_sample):
    """Create combined plot for transcripts and detected genes per cell distributions"""
    # Check if we have data for either transcripts or genes
    samples_with_transcript_counts = {}
    samples_with_gene_counts = {}

    for s_name, data in cells_data_by_sample.items():
        # Check for pre-calculated statistics first, fall back to raw values
        if data and "total_counts_box_stats" in data:
            samples_with_transcript_counts[s_name] = data["total_counts_box_stats"]
        elif data and "total_counts_values" in data and data["total_counts_values"]:
            samples_with_transcript_counts[s_name] = data["total_counts_values"]

        if data and "detected_genes_stats" in data:
            samples_with_gene_counts[s_name] = data["detected_genes_stats"]
        elif data and "detected_genes_values" in data and data["detected_genes_values"]:
            samples_with_gene_counts[s_name] = data["detected_genes_values"]

    # If neither dataset is available, return None
    if not samples_with_transcript_counts and not samples_with_gene_counts:
        return None

    num_samples = max(len(samples_with_transcript_counts), len(samples_with_gene_counts))

    if num_samples == 1:
        # Single sample: Create combined density plots
        return _create_single_sample_combined_density(samples_with_transcript_counts, samples_with_gene_counts)
    else:
        # Multiple samples: Create combined box plots
        return _create_multi_sample_combined_boxes(samples_with_transcript_counts, samples_with_gene_counts)


def _create_single_sample_combined_density(samples_with_transcript_counts, samples_with_gene_counts):
    """Create single sample combined density plot with transcripts (blue) and genes (grey) on the same plot"""
    plot_data = {}

    # Store raw values for intelligent line positioning
    raw_transcript_values = None
    raw_gene_values = None

    # Handle transcripts per cell data
    if samples_with_transcript_counts:
        _, transcript_values = next(iter(samples_with_transcript_counts.items()))
        # Skip density plots for pre-calculated statistics (use box plots instead)
        if isinstance(transcript_values, dict) and "min" in transcript_values:
            log.info(
                "Skipping density plot for transcripts per cell - using pre-calculated statistics. Density plots require raw data."
            )
            return None
        raw_transcript_values = transcript_values
        transcript_values = np.array(transcript_values)


        kde = stats.gaussian_kde(transcript_values)
        x_min, x_max = transcript_values.min(), transcript_values.max()
        x_range = np.linspace(x_min, x_max, 1000)
        density = kde(x_range)

        # Add to plot data
        transcripts_data = {}
        for x, y in zip(x_range, density):
            transcripts_data[float(x)] = float(y)
        plot_data["Transcripts per cell"] = transcripts_data

    # Handle detected genes per cell data
    if samples_with_gene_counts:
        _, gene_counts = next(iter(samples_with_gene_counts.items()))
        # Skip density plots for pre-calculated statistics
        if isinstance(gene_counts, dict) and "min" in gene_counts:
            log.info(
                "Skipping density plot for detected genes per cell - using pre-calculated statistics. Density plots require raw data."
            )
            # For mixed cases, only show available density plots
            if not raw_transcript_values:
                return None
        else:
            raw_gene_values = gene_counts

        gene_counts = np.array(gene_counts)

        kde = stats.gaussian_kde(gene_counts)
        x_min, x_max = gene_counts.min(), gene_counts.max()
        x_range = np.linspace(x_min, x_max, 1000)
        density = kde(x_range)

        # Add to plot data with dataset identifier
        genes_data = {}
        for x, y in zip(x_range, density):
            genes_data[float(x)] = float(y)
        plot_data["Detected genes per cell"] = genes_data


    if not plot_data:
        return None

    config = {
        "id": "xenium_cell_distributions_combined",
        "title": "Xenium: Distribution of Transcripts per Cell",
        "xlab": "Number per cell",
        "ylab": "Density",
        "smooth_points": 100,
    }

    # Add color configuration
    colors = {"Transcripts per cell": "#7cb5ec", "Detected genes per cell": "#434348"}
    config["colors"] = colors

    # Add all mean/median lines with intelligent overlap prevention
    combined_lines = _create_non_overlapping_combined_lines(
        transcript_values=raw_transcript_values, gene_values=raw_gene_values, plot_data=plot_data
    )
    if combined_lines:
        config["x_lines"] = combined_lines

    return linegraph.plot(plot_data, config)


def _create_multi_sample_combined_boxes(samples_with_transcript_counts, samples_with_genes_counts):
    """Create multi-sample combined box plots for transcripts and genes per cell using pre-calculated statistics"""

    plot_data = []
    data_labels = []

    # Add transcripts per cell data (prefer statistics over raw values)
    if samples_with_transcript_counts:
        transcripts_data = {}
        for s_name, transcript_counts_stats in samples_with_transcript_counts.items():
            transcripts_data[s_name] = transcript_counts_stats
        plot_data.append(transcripts_data)
        data_labels.append({"name": "Transcripts per Cell", "ylab": "Transcripts per cell"})

    # Add detected genes per cell data (prefer statistics over raw values)
    if samples_with_genes_counts:
        genes_data = {}
        for s_name, gene_count_stats in samples_with_genes_counts.items():
            genes_data[s_name] = gene_count_stats
        plot_data.append(genes_data)
        data_labels.append({"name": "Detected Genes per Cell", "ylab": "Detected genes per cell"})

    config = {
        "id": "xenium_cell_distributions_combined",
        "title": "Xenium: Distribution of Transcripts per Cell",
        "boxpoints": False,
        "xlab": "Transcripts per cell",
        "data_labels": data_labels,
    }

    return box.plot(plot_data, config)


def xenium_cell_area_distribution_plot(cells_data_by_sample):
    """Create cell area distribution plot - line plot for single sample, violin plots for multiple"""
    # Check which samples have cell area data
    samples_with_areas = []
    for s_name, data in cells_data_by_sample.items():
        # Accept either pre-calculated statistics or raw values
        if ("cell_area_box_stats" in data) or ("cell_area_values" in data and data["cell_area_values"]):
            samples_with_areas.append(s_name)

    if not samples_with_areas:
        return None

    num_samples = len(samples_with_areas)

    if num_samples == 1:
        # Single sample: Create line plot (density) with vertical lines for mean/median
        return _create_single_sample_area_density(cells_data_by_sample[samples_with_areas[0]])
    else:
        # Multiple samples: Create violin plots
        return _create_multi_sample_area_violins(cells_data_by_sample, samples_with_areas)


def _create_single_sample_area_density(cell_data):
    """Create density plot for single sample with mean/median lines"""

    # Skip density plots if only pre-calculated statistics are available
    if "cell_area_values" not in cell_data:
        log.info(
            "Skipping cell area density plot - using pre-calculated statistics. Density plots require raw data."
        )
        return None

    cell_areas = cell_data["cell_area_values"]
    if not cell_areas or len(cell_areas) < 10:
        return None

    # Create density estimation
    kde = stats.gaussian_kde(cell_areas)

    # Create x values for density curve
    min_area = min(cell_areas)
    max_area = max(cell_areas)
    x_range = max_area - min_area
    x_vals = np.linspace(max(0, min_area - 0.1 * x_range), max_area + 0.1 * x_range, 200)
    density_vals = kde(x_vals)

    # Prepare data for linegraph
    density_data = {}
    for x, y in zip(x_vals, density_vals):
        density_data[float(x)] = float(y)

    config: Dict[str, Any] = {
        "id": "xenium_cell_area_distribution",
        "title": "Xenium: Cell Area Distribution",
        "xlab": "Cell area",
        "ylab": "Density",
        "xsuffix": " μm²",
    }

    # Add vertical lines for mean and median
    if "cell_area_mean" in cell_data and "cell_area_median" in cell_data:
        density_keys = [float(k) for k in density_data.keys()]
        config["x_lines"] = _create_non_overlapping_labels(
            cell_data["cell_area_mean"],
            cell_data["cell_area_median"],
            precision=1,
            suffix=" μm²",
            data_min=min(density_keys),
            data_max=max(density_keys),
        )

    return linegraph.plot({"Density": density_data}, config)


def _create_multi_sample_area_violins(cells_data_by_sample, samples_with_areas):
    """Create box plots for multiple samples using pre-calculated statistics"""

    # For box plots, we now provide pre-calculated statistics instead of raw data
    data = {}

    for s_name in samples_with_areas:
        cell_data = cells_data_by_sample[s_name]
        if "cell_area_box_stats" in cell_data:
            # Use pre-calculated box plot statistics
            data[s_name] = cell_data["cell_area_box_stats"]

    if not data:
        return None

    config = {
        "id": "xenium_cell_area_distribution",
        "title": "Xenium: Cell Area Distribution",
        "xlab": "Cell area (μm²)",
        "boxpoints": False,
    }

    return box.plot(data, config)


def xenium_nucleus_rna_fraction_plot(cells_data_by_sample):
    """Create nucleus RNA fraction distribution plot - density for single sample, box plots for multiple"""
    # Check which samples have nucleus RNA fraction data
    samples_with_nucleus_data = []
    for s_name, data in cells_data_by_sample.items():
        if "nucleus_rna_fraction_box_stats" in data or (
            "nucleus_rna_fraction_values" in data and data["nucleus_rna_fraction_values"]
        ):
            samples_with_nucleus_data.append(s_name)

    if not samples_with_nucleus_data:
        return None

    num_samples = len(samples_with_nucleus_data)

    if num_samples == 1:
        # Single sample: Create density plot
        return _create_single_sample_nucleus_density(cells_data_by_sample[samples_with_nucleus_data[0]])
    else:
        # Multiple samples: Create box plots
        return _create_multi_sample_nucleus_boxes(cells_data_by_sample, samples_with_nucleus_data)


def _create_single_sample_nucleus_density(cell_data):
    """Create density plot for single sample nucleus RNA fractions"""

    # Skip density plots if only pre-calculated statistics are available
    if "nucleus_rna_fraction_values" not in cell_data:
        log.info(
            "Skipping nucleus RNA fraction density plot - using pre-calculated statistics. Density plots require raw data."
        )
        return None

    nucleus_fractions = cell_data["nucleus_rna_fraction_values"]
    if not nucleus_fractions:
        return None

    # Use a more appropriate range for nucleus RNA fractions (0 to 1)
    x_range = np.linspace(0, 1, 200)

    # Calculate kernel density estimation
    try:
        kde = stats.stats.gaussian_kde(nucleus_fractions)
        density = kde(x_range)
    except Exception:
        # Fallback to histogram if KDE fails
        hist, bin_edges = np.histogram(nucleus_fractions, bins=50, range=(0, 1), density=True)
        x_range = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = hist

    # Trim long tail: find cutoff where all values above X are below 1% of max
    max_density = np.max(density)
    threshold = max_density * 0.01  # 1% of max

    # Find the last point where density is above threshold
    last_significant_point = len(density) - 1
    for i in range(len(density) - 1, -1, -1):
        if density[i] >= threshold:
            last_significant_point = i
            break

    # Trim the data to only include up to the last significant point
    if last_significant_point < len(density) - 1:
        x_range = x_range[: last_significant_point + 1]
        density = density[: last_significant_point + 1]

    # Create the density plot data
    data = {}
    data["Nucleus RNA Fraction Density"] = {str(x): y for x, y in zip(x_range, density)}

    config = {
        "id": "xenium_nucleus_rna_fraction_single",
        "title": "Xenium: Fraction of Transcripts in Nucleus",
        "xlab": "Distribution of the fraction of transcripts found in the nucleus across cells",
        "ylab": "Density",
        "data_labels": [
            {"name": "Density", "ylab": "Density"},
        ],
    }

    # Add vertical lines for mean and median
    mean_fraction = np.nanmean(nucleus_fractions)
    median_fraction = np.nanmedian(nucleus_fractions)

    density_keys = [float(k) for k in data["Nucleus RNA Fraction Density"].keys()]
    config["x_lines"] = _create_non_overlapping_labels(
        mean_fraction,
        median_fraction,
        precision=3,
        data_min=min(density_keys),
        data_max=max(density_keys),
    )

    plot = linegraph.plot(data, config)

    return plot


def _create_multi_sample_nucleus_boxes(cells_data_by_sample, samples_with_nucleus_data):
    """Create box plots for multiple samples using pre-calculated statistics"""

    # For box plots, we now provide pre-calculated statistics instead of raw data
    data = {}

    for s_name in samples_with_nucleus_data:
        cell_data = cells_data_by_sample[s_name]
        if "nucleus_rna_fraction_box_stats" in cell_data:
            # Use pre-calculated box plot statistics
            data[s_name] = cell_data["nucleus_rna_fraction_box_stats"]
        elif "nucleus_rna_fraction_values" in cell_data:
            # Fallback to raw data if statistics not available (backward compatibility)
            nucleus_fractions = cell_data["nucleus_rna_fraction_values"]
            if nucleus_fractions:
                data[s_name] = [float(fraction) for fraction in nucleus_fractions]

    if not data:
        return None

    config = {
        "id": "xenium_nucleus_rna_fraction_multi",
        "title": "Xenium: Fraction of Transcripts in Nucleus",
        "xlab": "Distribution of the fraction of transcripts found in the nucleus across cells",
        "boxpoints": False,
    }

    return box.plot(data, config)


def xenium_nucleus_cell_area_ratio_plot(cells_data_by_sample):
    """Create nucleus-to-cell area ratio distribution plot - density for single sample, box plots for multiple"""
    # Check which samples have nucleus-to-cell area ratio data
    samples_with_ratio_data = []
    for s_name, data in cells_data_by_sample.items():
        if "nucleus_to_cell_area_ratio_box_stats" in data or (
            "nucleus_to_cell_area_ratio_values" in data and data["nucleus_to_cell_area_ratio_values"]
        ):
            samples_with_ratio_data.append(s_name)

    if not samples_with_ratio_data:
        return None

    num_samples = len(samples_with_ratio_data)

    if num_samples == 1:
        # Single sample: Create density plot
        return _create_single_sample_ratio_density(cells_data_by_sample[samples_with_ratio_data[0]])
    else:
        # Multiple samples: Create box plots
        return _create_multi_sample_ratio_boxes(cells_data_by_sample, samples_with_ratio_data)


def _create_single_sample_ratio_density(cell_data):
    """Create density plot for single sample nucleus-to-cell area ratios"""

    # Skip density plots if only pre-calculated statistics are available
    if "nucleus_to_cell_area_ratio_values" not in cell_data:
        log.info(
            "Skipping nucleus-to-cell area ratio density plot - using pre-calculated statistics. Density plots require raw data."
        )
        return None

    ratio_values = cell_data["nucleus_to_cell_area_ratio_values"]
    if not ratio_values:
        return None

    # Use a reasonable range for nucleus-to-cell area ratios (0 to 1.0)
    x_range = np.linspace(0, 1.0, 200)

    # Calculate kernel density estimation
    try:
        kde = stats.stats.gaussian_kde(ratio_values)
        density = kde(x_range)
    except Exception:
        # Fallback to histogram if KDE fails
        hist, bin_edges = np.histogram(ratio_values, bins=50, range=(0, 1.0), density=True)
        x_range = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = hist

    # Create the density plot data
    data = {}
    data["Nucleus-to-Cell Area Ratio Density"] = {str(x): y for x, y in zip(x_range, density)}

    config = {
        "id": "xenium_nucleus_cell_area_ratio_single",
        "title": "Xenium: Nucleus to Cell Area Distribution",
        "xlab": "Nucleus-to-cell area ratio",
        "ylab": "Density",
        "data_labels": [
            {"name": "Density", "ylab": "Density"},
        ],
    }

    # Add vertical lines for mean and median
    mean_ratio = np.nanmean(ratio_values)
    median_ratio = np.nanmedian(ratio_values)

    density_keys = [float(k) for k in data["Nucleus-to-Cell Area Ratio Density"].keys()]
    config["x_lines"] = _create_non_overlapping_labels(
        mean_ratio, median_ratio, precision=3, data_min=min(density_keys), data_max=max(density_keys)
    )

    plot = linegraph.plot(data, config)

    return plot


def _create_multi_sample_ratio_boxes(cells_data_by_sample, samples_with_ratio_data):
    """Create box plots for multiple samples using pre-calculated statistics"""

    # For box plots, we now provide pre-calculated statistics instead of raw data
    data = {}

    for s_name in samples_with_ratio_data:
        cell_data = cells_data_by_sample[s_name]
        if "nucleus_to_cell_area_ratio_box_stats" in cell_data:
            # Use pre-calculated box plot statistics
            data[s_name] = cell_data["nucleus_to_cell_area_ratio_box_stats"]
        elif "nucleus_to_cell_area_ratio_values" in cell_data:
            # Fallback to raw data if statistics not available (backward compatibility)
            ratio_values = cell_data["nucleus_to_cell_area_ratio_values"]
            if ratio_values:
                data[s_name] = [float(ratio) for ratio in ratio_values]

    if not data:
        return None

    config = box.BoxPlotConfig(
        id="xenium_nucleus_cell_area_ratio_multi",
        title="Xenium: Nucleus to Cell Area Distribution",
        xlab="Nucleus-to-cell area ratio",
        boxpoints=False,
        xmin=0,
        xmax=1,
    )

    return box.plot(data, config)


def _create_non_overlapping_labels(
    mean_value,
    median_value,
    mean_color="red",
    median_color="green",
    precision=0,
    suffix="",
    prefix="",
    threshold_percent=5,
    data_min=None,
    data_max=None,
):
    """
    Create vertical line configurations with non-overlapping labels when mean and median are close.

    Args:
        mean_value: Mean value for vertical line
        median_value: Median value for vertical line
        mean_color: Color for mean line (default: "red")
        median_color: Color for median line (default: "green")
        precision: Decimal places for value display
        suffix: Unit suffix to add to labels (e.g., " μm²")
        prefix: Prefix for labels (e.g., "Transcripts ", "Genes ")
        threshold_percent: If values are within this percentage of plot range, offset labels
        data_min: Minimum value of the underlying data range (optional)
        data_max: Maximum value of the underlying data range (optional)

    Returns:
        List of line configurations with appropriate label positioning
    """
    # Calculate plot range for scale-aware overlap detection
    if data_min is not None and data_max is not None:
        plot_range = data_max - data_min

        # If data range is too small, use mean/median range
        if plot_range == 0:
            plot_range = max(abs(mean_value - median_value), max(abs(mean_value), abs(median_value), 1))
    else:
        # Fall back to using mean/median values to estimate scale
        plot_range = max(abs(mean_value), abs(median_value), 1)

    # Calculate percentage difference relative to plot scale
    value_diff = abs(mean_value - median_value)
    range_percent_diff = (value_diff / plot_range) * 100

    # Format values according to precision
    if precision == 0:
        mean_str = f"{mean_value:.0f}"
        median_str = f"{median_value:.0f}"
    else:
        mean_str = f"{mean_value:.{precision}f}"
        median_str = f"{median_value:.{precision}f}"

    # Create base line configurations
    lines = [
        {
            "value": float(mean_value),
            "color": mean_color,
            "dash": "dash",
            "width": 2,
            "label": f"{prefix}Mean ({mean_str}{suffix})",
        },
        {
            "value": float(median_value),
            "color": median_color,
            "dash": "dash",
            "width": 2,
            "label": f"{prefix}Median ({median_str}{suffix})",
        },
    ]

    # If values are too close on the plot scale, create labels with non-breaking spaces to offset them horizontally
    if range_percent_diff < threshold_percent:
        # Use non-breaking spaces to create horizontal offset
        space = "&nbsp;" * 30
        lines[0]["label"] = f"{prefix}Mean ({mean_str}{suffix}){space}"  # Add trailing spaces
        lines[1]["label"] = f"{space}{prefix}Median ({median_str}{suffix})"  # Add leading spaces

    return lines


def _create_non_overlapping_combined_lines(
    transcript_values=None, gene_values=None, plot_data=None, threshold_percent=5
):
    """
    Create all vertical lines for combined plots with intelligent label positioning to avoid any overlaps.

    Args:
        transcript_values: Array of transcript values (optional)
        gene_values: Array of gene values (optional)
        plot_data: Dictionary of plot data to calculate X-axis range (optional)
        threshold_percent: Minimum percentage difference relative to plot range

    Returns:
        List of all line configurations with non-overlapping labels
    """
    lines = []
    all_values = []  # Track all line values for overlap detection

    # Collect transcript lines if provided
    if transcript_values is not None:
        mean_transcripts = np.nanmean(transcript_values)
        median_transcripts = np.nanmedian(transcript_values)

        transcript_lines = [
            {
                "value": float(mean_transcripts),
                "color": "#7cb5ec",
                "dash": "dash",
                "width": 2,
                "label": f"Transcripts Mean ({mean_transcripts:.0f})",
                "type": "mean",
                "dataset": "transcripts",
            },
            {
                "value": float(median_transcripts),
                "color": "#99c2e8",
                "dash": "dash",
                "width": 2,
                "label": f"Transcripts Median ({median_transcripts:.0f})",
                "type": "median",
                "dataset": "transcripts",
            },
        ]
        lines.extend(transcript_lines)
        all_values.extend([mean_transcripts, median_transcripts])

    # Collect gene lines if provided
    if gene_values is not None:
        mean_genes = np.nanmean(gene_values)
        median_genes = np.nanmedian(gene_values)

        gene_lines = [
            {
                "value": float(mean_genes),
                "color": "#434348",
                "dash": "dash",
                "width": 2,
                "label": f"Genes Mean ({mean_genes:.0f})",
                "type": "mean",
                "dataset": "genes",
            },
            {
                "value": float(median_genes),
                "color": "#888888",
                "dash": "dash",
                "width": 2,
                "label": f"Genes Median ({median_genes:.0f})",
                "type": "median",
                "dataset": "genes",
            },
        ]
        lines.extend(gene_lines)
        all_values.extend([mean_genes, median_genes])

    if not lines:
        return []

    # Sort lines by value for easier overlap detection
    lines.sort(key=lambda x: x["value"])

    # Calculate plot range from actual plot data X values
    if plot_data:
        all_x_values = []
        for dataset in plot_data.values():
            all_x_values.extend(dataset.keys())

        if all_x_values:
            min_value = min(all_x_values)
            max_value = max(all_x_values)
            plot_range = max_value - min_value
        else:
            # Fallback to line values if no plot data
            all_line_values = [line["value"] for line in lines]
            min_value = min(all_line_values)
            max_value = max(all_line_values)
            plot_range = max_value - min_value
    else:
        # Fallback to line values if no plot data provided
        all_line_values = [line["value"] for line in lines]
        min_value = min(all_line_values)
        max_value = max(all_line_values)
        plot_range = max_value - min_value

    # If plot range is too small, fall back to absolute threshold
    if plot_range == 0:
        plot_range = max(abs(max_value), 1)  # Avoid division by zero

    # Group overlapping lines and apply spacing once per group
    processed = set()

    for i in range(len(lines)):
        if i in processed:
            continue

        line = lines[i]
        overlap_group = [i]

        # Find all lines that overlap with this one
        for j in range(i + 1, len(lines)):
            if j in processed:
                continue

            other_line = lines[j]
            value_diff = abs(line["value"] - other_line["value"])

            # Calculate percentage relative to the plot range, not individual values
            range_percent_diff = (value_diff / plot_range) * 100

            if range_percent_diff < threshold_percent:
                overlap_group.append(j)

        # Apply spacing to the entire overlap group
        if len(overlap_group) > 1:
            space = "&nbsp;" * 15
            group_size = len(overlap_group)

            for idx, line_idx in enumerate(overlap_group):
                target_line = lines[line_idx]

                if group_size == 2:
                    # Two lines: one gets trailing space, other gets leading space
                    if idx == 0:
                        target_line["label"] = target_line["label"] + space
                    else:
                        target_line["label"] = space + target_line["label"]
                elif group_size == 3:
                    # Three lines: spread out with different amounts of spacing
                    if idx == 0:
                        target_line["label"] = target_line["label"] + space + space
                    elif idx == 1:
                        target_line["label"] = space + target_line["label"] + space
                    else:
                        target_line["label"] = space + space + target_line["label"]
                elif group_size >= 4:
                    # Four or more lines: maximum spreading
                    if idx == 0:
                        target_line["label"] = target_line["label"] + space + space + space
                    elif idx == 1:
                        target_line["label"] = target_line["label"] + space
                    elif idx == group_size - 2:
                        target_line["label"] = space + target_line["label"]
                    else:
                        target_line["label"] = space + space + space + target_line["label"]

                processed.add(line_idx)

    # Clean up temporary fields
    for line in lines:
        line.pop("type", None)
        line.pop("dataset", None)

    return lines
