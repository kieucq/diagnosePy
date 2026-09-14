"""Create multi-year monthly averages from ERA5 NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "era5" / "1946_1960"
ERA5_FILENAME_REGEX = re.compile(r"ERA5_sfc_(\d{4})(\d{2})(\d{2})T\d{6}\.\d+\.nc$")
TIME_DIMS = {"time", "valid_time"}
LEVEL_DIMS = ("level", "pressure_level", "isobaricInhPa", "isobaricInPa", "plev")


def _xr():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to read ERA5 NetCDF files. Install xarray in "
            "the Python environment where you call create_era5_average."
        ) from exc
    return xr


def _normalize_month(month):
    month = int(month)
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 1 and 12; got {month}")
    return month


def _era5_file_date(path):
    match = ERA5_FILENAME_REGEX.match(Path(path).name)
    if not match:
        return None
    year, month, day = match.groups()
    return int(year), int(month), int(day)


def _resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if data_dir.exists() or data_dir.is_absolute():
        return data_dir
    project_relative = PROJECT_ROOT / data_dir
    if project_relative.exists():
        return project_relative
    return data_dir


def _matching_era5_files(data_dir, start_year, end_year, month):
    data_dir = _resolve_data_dir(data_dir)
    month = _normalize_month(month)
    if int(end_year) < int(start_year):
        raise ValueError("end_year must be greater than or equal to start_year.")
    if not data_dir.exists():
        raise FileNotFoundError(f"ERA5 data directory was not found: {data_dir}")

    files = []
    for path in data_dir.glob("ERA5_sfc_*.nc"):
        file_date = _era5_file_date(path)
        if file_date is None:
            continue
        year, file_month, _day = file_date
        if int(start_year) <= year <= int(end_year) and file_month == month:
            files.append(path)
    return sorted(files)


def _level_label(level, selected_level=None):
    if selected_level is not None:
        value = float(selected_level)
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    if level is None:
        return "sfc"
    text = str(level).strip()
    return text.replace("/", "-").replace(" ", "")


def _coerce_level(level):
    if level is None:
        return None
    if isinstance(level, str):
        text = level.strip()
        if text.lower() in ("", "none", "sfc", "surface"):
            return None
        return float(text)
    return float(level)


def _select_level(data_array, level):
    level_dim = next((dim for dim in LEVEL_DIMS if dim in data_array.dims), None)
    requested_level = _coerce_level(level)
    if level_dim is None:
        if requested_level is not None:
            warnings.warn(
                f"Variable {data_array.name!r} has no level dimension; ignoring level={level!r}.",
                RuntimeWarning,
            )
        return data_array, None
    if requested_level is None:
        raise ValueError(
            f"Variable {data_array.name!r} has level dimension {level_dim!r}; "
            "please provide a level such as 850."
        )

    selected = data_array.sel({level_dim: requested_level}, method="nearest")
    selected_level = float(np.asarray(selected[level_dim].values).ravel()[0])
    return selected, selected_level


def _sum_and_count(data_array):
    time_dims = [dim for dim in data_array.dims if dim.lower() in TIME_DIMS]
    if time_dims:
        summed = data_array.sum(dim=time_dims, skipna=True)
        counted = data_array.notnull().sum(dim=time_dims)
    else:
        summed = data_array.where(data_array.notnull(), 0)
        counted = data_array.notnull()
    return np.asarray(summed.values, dtype=np.float64), np.asarray(counted.values, dtype=np.int64)


def _default_output_path(output_dir, start_year, end_year, month, variable, level_label):
    return Path(output_dir) / f"era5_{int(start_year)}_{int(end_year)}_{int(month):02d}_{variable}_{level_label}.npy"


def create_era5_average(
    start_year,
    end_year,
    month,
    variable,
    level=None,
    data_dir=DEFAULT_DATA_DIR,
    output_dir=".",
    output_file=None,
    skip_missing_variable=False,
    verbose=False,
    progress_interval=50,
    return_data=False,
):
    """
    Create a multi-year monthly average for one ERA5 variable and save it as NPY.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range. A 10-year average is obtained by passing a
        10-year inclusive range, for example ``1946, 1955``.
    month : int
        Month to average, 1 through 12.
    variable : str
        ERA5 variable name in the NetCDF files, for example ``MSL`` or ``T``.
    level : int, float, str, or None
        Pressure level to select for level-dependent variables. Use ``None`` or
        ``"sfc"`` for surface variables.
    data_dir : str or Path
        Directory containing files named like
        ``ERA5_sfc_yyyymmddT000000.000000000.nc``.
    output_dir, output_file : str or Path
        Output location. If ``output_file`` is omitted, the filename is
        ``era5_{start_year}_{end_year}_{month}_{variable}_{level}.npy``.
    skip_missing_variable : bool
        If True, ignore files that do not contain ``variable``.
    return_data : bool
        If True, return ``(average, output_path, metadata)``. Otherwise return
        only ``output_path``.
    """
    month = _normalize_month(month)
    files = _matching_era5_files(data_dir, start_year, end_year, month)
    if not files:
        raise FileNotFoundError(
            f"No ERA5 files found in {Path(data_dir)} for years {start_year}-{end_year}, month {month:02d}."
        )
    if verbose:
        print(
            f"Found {len(files)} ERA5 files for years {int(start_year)}-{int(end_year)}, "
            f"month {month:02d}, variable {variable}, level {_level_label(level)}."
        )

    total = None
    count = None
    selected_level = None
    used_files = []
    variable_dims = None

    for file_index, path in enumerate(files, start=1):
        with _xr().open_dataset(path) as dataset:
            if variable not in dataset:
                if skip_missing_variable:
                    continue
                available = ", ".join(list(dataset.data_vars)[:30])
                raise KeyError(f"Variable {variable!r} was not found in {path}. First available variables: {available}")

            data_array, file_selected_level = _select_level(dataset[variable], level)
            if selected_level is None and file_selected_level is not None:
                selected_level = file_selected_level
            file_sum, file_count = _sum_and_count(data_array)

        if total is None:
            total = np.zeros_like(file_sum, dtype=np.float64)
            count = np.zeros_like(file_count, dtype=np.int64)
            variable_dims = tuple(dim for dim in data_array.dims if dim.lower() not in TIME_DIMS)
        elif total.shape != file_sum.shape:
            raise ValueError(f"Shape mismatch while reading {path}: expected {total.shape}, got {file_sum.shape}")

        total += file_sum
        count += file_count
        used_files.append(path)
        if verbose and progress_interval and (
            file_index == 1 or file_index == len(files) or file_index % int(progress_interval) == 0
        ):
            print(f"Processed {file_index}/{len(files)}: {path.name}")

    if total is None or not used_files:
        raise KeyError(f"Variable {variable!r} was not found in any selected ERA5 file.")

    average = np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float64), where=count > 0)
    level_name = _level_label(level, selected_level)
    output_path = Path(output_file) if output_file is not None else _default_output_path(
        output_dir, start_year, end_year, month, variable, level_name
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, average)
    if verbose:
        print(f"Wrote {output_path}")

    metadata = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "month": month,
        "variable": variable,
        "level": level_name,
        "selected_level": selected_level,
        "data_dir": str(Path(data_dir)),
        "output_file": str(output_path),
        "file_count": len(used_files),
        "sample_count_min": int(np.nanmin(count)),
        "sample_count_max": int(np.nanmax(count)),
        "shape": average.shape,
        "dims": variable_dims,
    }
    if return_data:
        return average, output_path, metadata
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a multi-year monthly ERA5 average and save it as a .npy file.")
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("variable")
    parser.add_argument("level", nargs="?", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--skip-missing-variable", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args(argv)

    output_path = create_era5_average(
        args.start_year,
        args.end_year,
        args.month,
        args.variable,
        args.level,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_file=args.output_file,
        skip_missing_variable=args.skip_missing_variable,
        verbose=not args.quiet,
        progress_interval=args.progress_interval,
    )
    print(output_path)


if __name__ == "__main__":
    main()
