# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MultiQC plugin that extends the core Xenium module with computationally intensive analyses for Xenium spatial transcriptomics data. It processes large parquet and H5 files that the core module skips for performance reasons.

## Development Commands

### Install for development

```bash
pip install -e .
```

### Run linting

```bash
pre-commit run --all-files
```

Uses ruff for formatting and linting (line-length: 120, target: py39).

### Run MultiQC with the plugin

```bash
multiqc /path/to/xenium/data
```

## Architecture

The plugin uses MultiQC's hook system (v1) to extend functionality:

- **`xenium_extra_execution_start`**: Called at startup to register search patterns for `transcripts.parquet`, `cells.parquet`, and `cell_feature_matrix.h5` files, and increase file size limit to 5GB.

- **`extend_xenium_module`**: Called after the core Xenium module runs, receives the module instance and adds extra sections/data.

All plugin code is in `multiqc_xenium_extra/xenium_extra.py`. The extension function:

1. Parses parquet files using polars and H5 files using scanpy
2. Merges computed metrics (cell area, nucleus area) into the module's `data_by_sample` dict
3. Adds new general stats columns via `xenium_module.genstat_headers`
4. Adds report sections via `xenium_module.add_section()`

## Key Dependencies

- **polars**: For efficient parquet file parsing
- **scanpy**: For H5 matrix analysis
- **scipy**: For statistical computations (KDE, distributions)

## Gene Categories

The plugin categorizes features by prefix:

- `Custom_` → Custom
- `NegControlProbe_` → Negative Control Probe
- `NegControlCodeword_` → Negative Control Codeword
- `GenomicControlProbe_` → Genomic Control Probe
- `UnassignedCodeword_` → Unassigned Codeword
- Default → Pre-designed
