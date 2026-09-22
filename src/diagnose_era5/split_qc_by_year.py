"""Split the ERA5 QC NetCDF file into period-specific NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "era5" / "1980_2020_w" / "w_1980_2022.nc"


def _xr():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to split the ERA5 QC NetCDF file. Load or install "
            "xarray in the Python environment used for this script."
        ) from exc
    return xr


def parse_period(period):
    """Parse a period string such as 1980-1985, 1980_1985, or 1980:1985."""
    match = re.fullmatch(r"\s*(\d{4})\D+(\d{4})\s*", str(period))
    if match is None:
        raise ValueError(f"Period must look like 1980-1985; got {period!r}")
    start_year, end_year = (int(value) for value in match.groups())
    if end_year < start_year:
        raise ValueError(f"Period end year must be >= start year; got {period!r}")
    return start_year, end_year


def _find_time_dim(dataset):
    for name in ("valid_time", "time"):
        if name in dataset.dims:
            return name
    raise ValueError(f"Could not find a time dimension. Dataset dims are {tuple(dataset.dims)}")


def _period_slice(start_year, end_year):
    return slice(f"{int(start_year):04d}-01-01", f"{int(end_year):04d}-12-31T23:59:59")


def _output_path(output_dir, start_year, end_year):
    return Path(output_dir) / f"w_{int(start_year)}_{int(end_year)}.nc"


def _time_count(dataset, time_dim):
    return int(dataset.sizes.get(time_dim, 0))


def _encoding_for_output(dataset):
    """Preserve practical compression settings for data variables."""
    encoding = {}
    for name in dataset.data_vars:
        variable = dataset[name]
        if variable.ndim == 0:
            continue

        source_encoding = variable.encoding
        variable_encoding = {
            "zlib": bool(source_encoding.get("zlib", True)),
            "complevel": int(source_encoding.get("complevel", 1)),
            "shuffle": bool(source_encoding.get("shuffle", True)),
        }
        if "dtype" in source_encoding and source_encoding["dtype"] is not None:
            variable_encoding["dtype"] = source_encoding["dtype"]
        if "_FillValue" in source_encoding:
            variable_encoding["_FillValue"] = source_encoding["_FillValue"]
        if source_encoding.get("chunksizes") is not None:
            variable_encoding["chunksizes"] = tuple(
                min(int(chunk), int(size))
                for chunk, size in zip(source_encoding["chunksizes"], variable.shape)
            )
        encoding[name] = variable_encoding
    return encoding


def split_qc_file(
    input_file,
    period,
    *,
    output_dir=None,
    overwrite=False,
    dry_run=False,
):
    """Write one period-specific QC NetCDF file and return its path."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input QC file was not found: {input_file}")

    start_year, end_year = parse_period(period)
    output_dir = Path(output_dir) if output_dir is not None else input_file.parent
    output_path = _output_path(output_dir, start_year, end_year)
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    xr = _xr()
    with xr.open_dataset(input_file) as dataset:
        time_dim = _find_time_dim(dataset)
        subset = dataset.sel({time_dim: _period_slice(start_year, end_year)})
        count = _time_count(subset, time_dim)
        if count == 0:
            raise ValueError(f"No {time_dim} records found for {start_year}-{end_year}.")

        print(f"{start_year}-{end_year}: selected {count} {time_dim} records -> {output_path}", flush=True)
        if dry_run:
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        subset.to_netcdf(tmp_path, encoding=_encoding_for_output(subset))
        tmp_path.replace(output_path)

    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Split data/era5/1980_2020_w/w_1980_2022.nc into two period files. "
            "Example: python split_w_periods.py 1980-1985 2017-2021"
        )
    )
    parser.add_argument("period1", help="First period, for example 1980-1985.")
    parser.add_argument("period2", help="Second period, for example 2017-2021.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), help="Input QC NetCDF file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for split NetCDF files. Defaults to the input file directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected record counts without writing files.")
    args = parser.parse_args(argv)

    outputs = []
    for period in (args.period1, args.period2):
        outputs.append(
            split_qc_file(
                args.input_file,
                period,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run:
        for output in outputs:
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
