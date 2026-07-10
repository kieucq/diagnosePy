# Data

This directory stores small shareable inputs and local links to large WRF model output.

## Contents

- `best_tracks/bwp122024.dat`: JTWC/ATCF best-track file for Typhoon Yagi, WP122024.
- `wrfout/`: local WRF output files or symlinks used by the notebooks. These files are intentionally ignored by git because WRF NetCDF outputs are usually large.

## WRF File Naming

The workflow utilities sort WRF files by timestamps embedded in names like:

```text
wrfout_d01_2024-09-06_12:00:00
wrfout_d03_2024-09-06_18:00:00
```

To run the notebooks on another system, place or symlink the required `wrfout_*`, `wrfinput_*`, or `wrfbdy_*` files in `data/wrfout/`.
