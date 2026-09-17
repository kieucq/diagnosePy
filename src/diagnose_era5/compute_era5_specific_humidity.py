#!/usr/bin/env python3
"""Compute ERA5 specific humidity from averaged R and T NPY files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


ERA5_AVERAGE_RE = re.compile(
    r"^(?P<head>era5_\d{4}_\d{4}_\d{2})_(?P<variable>[A-Za-z0-9]+)_(?P<level>\d+(?:\.\d+)?)\.npy$"
)


def parse_era5_average_filename(path):
    """Return metadata from a filename like era5_1980_1985_09_R_950.npy."""
    path = Path(path)
    match = ERA5_AVERAGE_RE.match(path.name)
    if match is None:
        raise ValueError(
            "Expected a filename like era5_1980_1985_09_R_950.npy; "
            f"got {path.name!r}"
        )
    metadata = match.groupdict()
    metadata["level_hpa"] = float(metadata["level"])
    metadata["path"] = path
    return metadata


def identify_r_and_t_files(file1, file2):
    """Identify the relative humidity and temperature files from two inputs."""
    meta1 = parse_era5_average_filename(file1)
    meta2 = parse_era5_average_filename(file2)
    by_variable = {meta1["variable"].upper(): meta1, meta2["variable"].upper(): meta2}
    if set(by_variable) != {"R", "T"}:
        raise ValueError(
            "The two input files must contain one R file and one T file; "
            f"got variables {meta1['variable']!r} and {meta2['variable']!r}."
        )

    r_meta = by_variable["R"]
    t_meta = by_variable["T"]
    if r_meta["head"] != t_meta["head"]:
        raise ValueError(
            "The R and T filenames do not describe the same average period/month: "
            f"{r_meta['head']!r} vs {t_meta['head']!r}."
        )
    if not np.isclose(r_meta["level_hpa"], t_meta["level_hpa"]):
        raise ValueError(
            "The R and T pressure levels do not match: "
            f"{r_meta['level_hpa']:g} hPa vs {t_meta['level_hpa']:g} hPa."
        )
    return r_meta, t_meta


def load_average_field(path):
    """Load a 2-D averaged ERA5 field from a NumPy output file."""
    field = np.squeeze(np.load(path))
    if field.ndim != 2:
        raise ValueError(f"Expected a 2-D field after squeeze, got shape {field.shape}")
    return field


def specific_humidity_from_rh_temperature(relative_humidity_percent, temperature_k, pressure_hpa):
    """Compute specific humidity in kg kg-1 from RH percent, T in K, and pressure in hPa."""
    rh = np.asarray(relative_humidity_percent, dtype=float)
    temperature_k = np.asarray(temperature_k, dtype=float)
    if rh.shape != temperature_k.shape:
        raise ValueError(f"R and T arrays must have the same shape; got {rh.shape} and {temperature_k.shape}")

    rh_fraction = np.clip(rh / 100.0, 0.0, 1.0)
    temperature_c = temperature_k - 273.15
    saturation_vapor_pressure_hpa = 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    vapor_pressure_hpa = rh_fraction * saturation_vapor_pressure_hpa

    epsilon = 0.622
    denominator = float(pressure_hpa) - (1.0 - epsilon) * vapor_pressure_hpa
    return epsilon * vapor_pressure_hpa / denominator


def output_path_for_specific_humidity(r_meta):
    """Build the Q output path beside the R file."""
    level = r_meta["level"]
    output_name = f"{r_meta['head']}_Q_{level}.npy"
    return r_meta["path"].resolve().parent / output_name


def compute_specific_humidity_file(file1, file2):
    """Compute and save Q from two ERA5 average files."""
    r_meta, t_meta = identify_r_and_t_files(file1, file2)
    relative_humidity = load_average_field(r_meta["path"])
    temperature_k = load_average_field(t_meta["path"])
    specific_humidity = specific_humidity_from_rh_temperature(
        relative_humidity,
        temperature_k,
        r_meta["level_hpa"],
    )

    output_path = output_path_for_specific_humidity(r_meta)
    np.save(output_path, specific_humidity)
    return output_path, specific_humidity, r_meta["level_hpa"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute ERA5 specific humidity Q from one averaged relative humidity "
            "file and one averaged temperature file. The pressure level is parsed "
            "from the filenames, and the output is saved beside the inputs."
        )
    )
    parser.add_argument("file1", help="ERA5 averaged R or T .npy file")
    parser.add_argument("file2", help="ERA5 averaged T or R .npy file")
    args = parser.parse_args(argv)

    output_path, specific_humidity, pressure_hpa = compute_specific_humidity_file(args.file1, args.file2)
    print(f"Pressure level: {pressure_hpa:g} hPa")
    print(f"Q shape: {specific_humidity.shape}")
    print(
        "Q range: "
        f"{np.nanmin(specific_humidity):.6g} to {np.nanmax(specific_humidity):.6g} kg kg-1"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
