# multiqc-xenium-extra changelog

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
