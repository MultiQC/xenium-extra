# MultiQC Xenium Extra Plugin

Advanced analysis plugin for Xenium spatial transcriptomics data in MultiQC.

This plugin extends the core MultiQC Xenium module with computationally intensive analyses including:

- Parquet file processing (transcripts.parquet, cells.parquet)
- H5 matrix analysis (cell_feature_matrix.h5)
- Distribution plots (density, violin, box plots)
- Cell area and nucleus analysis
- Transcript quality distributions
- FoV quality heatmaps

## Installation

```bash
pip install multiqc-xenium-extra
```

## Requirements

- MultiQC >= 1.20
- polars >= 0.18.0
- scipy >= 1.8.0
- scanpy >= 1.9.0

## Configuration

The plugin automatically adjusts the file size limit to handle large Xenium files:

```yaml
log_filesize_limit: 5000000000  # 5GB
```

## Usage

Once installed, the plugin automatically extends the Xenium module when you run MultiQC:

```bash
multiqc /path/to/xenium/data
```

The extra sections will appear alongside the core Xenium module sections in the report.
