"""Average ERA5 cloud liquid water content for fixed climate periods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = Path("/N/u/ckieu/BigRed200/codex/diagnosePy/data/era5/1980_2020_qc/qc_1980_2022.nc")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PERIODS = ((1980, 1985), (2017, 2021))
TIME_DIMS = ("valid_time", "time")
LEVEL_DIMS = ("pressure_level", "level", "isobaricInhPa", "isobaricInPa", "plev")


def _xr():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to read ERA5 NetCDF files. Load or install xarray "
            "in the Python environment used for this script."
        ) from exc
    return xr


def _normalize_month(month):
    month = int(month)
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 1 and 12; got {month}")
    return month


def _level_label(level):
    value = float(level)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _find_time_dim(data_array):
    for dim in TIME_DIMS:
        if dim in data_array.dims:
            return dim
    raise ValueError(f"Could not find a time dimension in {data_array.name!r}; dims are {data_array.dims}")


def _select_pressure_level(data_array, level_hpa):
    level_dim = next((dim for dim in LEVEL_DIMS if dim in data_array.dims), None)
    if level_dim is None:
        return data_array, None

    coord = data_array[level_dim]
    target_level = float(level_hpa)
    units = str(coord.attrs.get("units", "")).lower()
    if level_dim == "isobaricInPa" or units in ("pa", "pascal", "pascals"):
        target_value = target_level * 100.0
        selected_level_hpa = lambda value: float(value) / 100.0
    else:
        target_value = target_level
        selected_level_hpa = float

    selected = data_array.sel({level_dim: target_value}, method="nearest")
    selected_value = np.asarray(selected[level_dim].values).ravel()[0]
    selected_hpa = selected_level_hpa(selected_value)
    if not np.isclose(selected_hpa, target_level, atol=0.01):
        raise ValueError(f"Nearest pressure level is {selected_hpa:g} hPa, not requested {target_level:g} hPa.")
    return selected, selected_hpa


def _time_indices_for_period_month(time_values, start_year, end_year, month):
    dates = np.asarray(time_values).astype("datetime64[D]")
    date_strings = np.datetime_as_string(dates, unit="D")
    years = np.array([int(value[:4]) for value in date_strings], dtype=np.int32)
    months = np.array([int(value[5:7]) for value in date_strings], dtype=np.int16)
    mask = (years >= int(start_year)) & (years <= int(end_year)) & (months == int(month))
    return np.where(mask)[0]


def _contiguous_index_ranges(indices):
    """Yield contiguous integer index ranges as (start, stop) pairs."""
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    for group in np.split(indices, breaks):
        yield int(group[0]), int(group[-1]) + 1


def _output_path(output_dir, start_year, end_year, month, output_variable, level_hpa):
    return (
        Path(output_dir)
        / f"era5_{int(start_year)}_{int(end_year)}_{int(month):02d}_{output_variable}_{_level_label(level_hpa)}.npy"
    )


def average_cloud_water_period(
    data_file,
    start_year,
    end_year,
    month,
    *,
    variable="clwc",
    level_hpa=950.0,
    batch_size=31,
    verbose=False,
):
    """Average one ERA5 cloud-water variable for one month over a year range."""
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"ERA5 cloud-water file was not found: {data_file}")
    month = _normalize_month(month)
    if int(end_year) < int(start_year):
        raise ValueError("end_year must be greater than or equal to start_year.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1.")

    with _xr().open_dataset(data_file) as dataset:
        if variable not in dataset:
            available = ", ".join(list(dataset.data_vars))
            raise KeyError(f"Variable {variable!r} was not found in {data_file}. Available variables: {available}")

        data_array, selected_level = _select_pressure_level(dataset[variable], level_hpa)
        time_dim = _find_time_dim(data_array)
        time_indices = _time_indices_for_period_month(data_array[time_dim].values, start_year, end_year, month)
        if time_indices.size == 0:
            raise ValueError(f"No times found for {start_year}-{end_year}, month {month:02d}.")

        total = None
        count = None
        processed = 0
        for range_start, range_stop in _contiguous_index_ranges(time_indices):
            for batch_start in range(range_start, range_stop, int(batch_size)):
                batch_stop = min(batch_start + int(batch_size), range_stop)
                if verbose:
                    print(
                        f"  reading {start_year}-{end_year} month {month:02d}: "
                        f"records {processed + 1}-{processed + (batch_stop - batch_start)} "
                        f"of {time_indices.size}",
                        flush=True,
                    )
                batch = data_array.isel({time_dim: slice(batch_start, batch_stop)})
                batch_sum = batch.sum(dim=time_dim, skipna=True)
                batch_count = batch.notnull().sum(dim=time_dim)

                batch_sum_values = np.asarray(batch_sum.values, dtype=np.float64)
                batch_count_values = np.asarray(batch_count.values, dtype=np.int64)
                if total is None:
                    total = np.zeros_like(batch_sum_values, dtype=np.float64)
                    count = np.zeros_like(batch_count_values, dtype=np.int64)
                total += batch_sum_values
                count += batch_count_values
                processed += batch_stop - batch_start

    average = np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float64), where=count > 0)
    return np.squeeze(average), {
        "data_file": str(data_file),
        "variable": variable,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "month": month,
        "level_hpa": float(selected_level if selected_level is not None else level_hpa),
        "time_count": int(time_indices.size),
        "shape": tuple(np.squeeze(average).shape),
    }


def create_cloud_water_averages(
    month,
    *,
    data_file=DEFAULT_DATA_FILE,
    output_dir=DEFAULT_OUTPUT_DIR,
    periods=DEFAULT_PERIODS,
    variable="clwc",
    output_variable="QC",
    level_hpa=950.0,
    batch_size=31,
    verbose=True,
):
    """Create cloud-water averages for each requested period and save them as NPY files."""
    output_paths = []
    for start_year, end_year in periods:
        average, metadata = average_cloud_water_period(
            data_file,
            start_year,
            end_year,
            month,
            variable=variable,
            level_hpa=level_hpa,
            batch_size=batch_size,
            verbose=verbose,
        )
        output_path = _output_path(output_dir, start_year, end_year, month, output_variable, metadata["level_hpa"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, average)
        output_paths.append(output_path)
        if verbose:
            print(
                f"Wrote {output_path} from {metadata['time_count']} times; "
                f"range {np.nanmin(average):.6g} to {np.nanmax(average):.6g}"
            )
    return output_paths


def _parse_periods(period_args):
    if not period_args:
        return DEFAULT_PERIODS
    return tuple((int(start), int(end)) for start, end in period_args)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Average ERA5 specific cloud liquid water content for 1980-1985 and "
            "2017-2021 for a selected month."
        )
    )
    parser.add_argument("month", type=int, help="Month to average, 1-12.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE), help="Input NetCDF file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output .npy files.")
    parser.add_argument("--variable", default="clwc", help="Input NetCDF variable name.")
    parser.add_argument(
        "--output-variable",
        default="QC",
        help="Variable tag to use in output filenames. Defaults to QC to match the requested naming convention.",
    )
    parser.add_argument("--level", type=float, default=950.0, help="Pressure level in hPa.")
    parser.add_argument("--batch-size", type=int, default=31, help="Number of time records to read per batch.")
    parser.add_argument(
        "--period",
        nargs=2,
        action="append",
        metavar=("START_YEAR", "END_YEAR"),
        help="Optional period to average. Can be used more than once. Defaults to 1980 1985 and 2017 2021.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    args = parser.parse_args(argv)

    output_paths = create_cloud_water_averages(
        args.month,
        data_file=args.data_file,
        output_dir=args.output_dir,
        periods=_parse_periods(args.period),
        variable=args.variable,
        output_variable=args.output_variable,
        level_hpa=args.level,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )
    if args.quiet:
        for path in output_paths:
            print(path)


if __name__ == "__main__":
    main()
