"""
Reusable plotting workflows extracted from the WRF notebooks in this folder.

Main functions
--------------
plot_animation_d03
    Build a Matplotlib animation of one shaded variable over a list of WRF/NetCDF
    files.
    
plot_track
    Track and plot Typhoon using 850-hPa perturbation height, pressure over
    land, maximum 10-m wind, and a 50-km-per-6-hour continuity check.

plot_vertical_cross_section
    Plot a WRF vertical cross section with shading, signed contours, and
    vertical-plane wind vectors.
"""

from pathlib import Path
import csv
import re
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap

import modColorCode as color


DEFAULT_TIME_REGEX = r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"
DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def _xr():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to open WRF/NetCDF files. Install xarray in "
            "the Python environment where you call these plotting functions."
        ) from exc
    return xr


def _wrfplot():
    import modPlotWRF as wrfplot

    return wrfplot


def _as_paths(files):
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(item) for item in files]


def _parse_filename_time(path, regex=DEFAULT_TIME_REGEX, time_format=DEFAULT_TIME_FORMAT):
    match = re.search(regex, Path(path).name)
    if not match:
        return None, Path(path).name
    text = match.group(1)
    if time_format:
        try:
            return datetime.strptime(text, time_format), text
        except ValueError:
            pass
    return None, text


def _sort_files(files, regex=DEFAULT_TIME_REGEX, time_format=DEFAULT_TIME_FORMAT):
    def key(path):
        parsed_time, text = _parse_filename_time(path, regex, time_format)
        if parsed_time is not None:
            return (0, parsed_time, Path(path).name)
        return (1, text, Path(path).name)

    return sorted(_as_paths(files), key=key)


def _file_time_label(path, regex=DEFAULT_TIME_REGEX, time_format=DEFAULT_TIME_FORMAT):
    parsed_time, text = _parse_filename_time(path, regex, time_format)
    if parsed_time is not None:
        return parsed_time.strftime("%Y-%m-%d %H:%M:%S")
    return text


def _select_time_if_present(data_array, time_index=0):
    for dim in ("Time", "time", "times", "valid_time"):
        if dim in data_array.dims:
            return data_array.isel({dim: time_index})
    return data_array


def _wrf_time_label(dataset, time_index=0, fallback=None):
    if "Times" in dataset:
        raw = dataset["Times"].isel({dataset["Times"].dims[0]: time_index}).values
        flat = np.asarray(raw).ravel()
        if flat.dtype.kind == "S":
            return b"".join(flat).decode("utf-8", errors="ignore").strip()
        return "".join(str(item) for item in flat).strip()
    return fallback or f"frame {time_index + 1}"


def _lat_lon_arrays(dataset, time_index=0):
    if "XLAT" not in dataset or "XLONG" not in dataset:
        raise KeyError("WRF XLAT and XLONG variables are required.")
    lat = _select_time_if_present(dataset["XLAT"], time_index).squeeze(drop=True)
    lon = _select_time_if_present(dataset["XLONG"], time_index).squeeze(drop=True)
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError(f"Expected 2D XLAT/XLONG after time selection; got {lat.dims} and {lon.dims}")
    return lat.values, lon.values


def _drop_time_like(coord):
    out = coord
    for dim in list(out.dims):
        if dim in ("Time", "time", "times", "valid_time"):
            out = out.isel({dim: 0})
    return out


def _guess_time_dim(data_array):
    for name in ("Time", "time", "times", "valid_time"):
        if name in data_array.dims:
            return name
    return None


def _guess_xy_dims(data_array):
    candidates = [
        ("south_north", "west_east"),
        ("south_north_stag", "west_east"),
        ("south_north", "west_east_stag"),
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("y", "x"),
    ]
    for ydim, xdim in candidates:
        if ydim in data_array.dims and xdim in data_array.dims:
            return ydim, xdim
    if len(data_array.dims) >= 2:
        return data_array.dims[-2], data_array.dims[-1]
    raise ValueError(f"Could not infer horizontal dimensions from {data_array.dims}")


def _guess_level_dim(data_array, time_dim, ydim, xdim):
    vertical_names = (
        "bottom_top",
        "bottom_top_stag",
        "level",
        "lev",
        "z",
        "height",
        "pressure",
        "isobaric",
        "isobaricInhPa",
        "altitude",
    )
    for name in vertical_names:
        if name in data_array.dims:
            return name
    used = {dim for dim in (time_dim, ydim, xdim) if dim is not None}
    leftovers = [dim for dim in data_array.dims if dim not in used]
    return leftovers[0] if leftovers else None


def _normalize_bounds(start, end, size):
    start = 0 if start is None else int(start)
    end = size if end is None else int(end)
    start = max(0, min(start, size - 1))
    end = max(start + 1, min(end, size))
    return start, end


def _indices_from_latlon(dataset, slat, elat, slon, elon):
    lats, lons = _lat_lon_arrays(dataset)
    lat1d = lats[:, 0]
    lon1d = lons[0, :]
    rs = int(np.abs(slat - lat1d).argmin())
    re = int(np.abs(elat - lat1d).argmin())
    cs = int(np.abs(slon - lon1d).argmin())
    ce = int(np.abs(elon - lon1d).argmin())
    rs, re = sorted((rs, re))
    cs, ce = sorted((cs, ce))
    return rs, re + 1, cs, ce + 1


def _levels_and_cmap(
    values,
    levels=None,
    level_count=21,
    cmap_name="viridis",
    min_color=(0.9, 0.9, 1.0),
    max_color=(0.0, 0.0, 1.0),
    neg_color=(0.0, 0.0, 1.0),
    pos_color=(1.0, 0.0, 0.0),
    percentiles=(2, 98),
):
    if levels is None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("No finite values were found for automatic level generation.")
        vmin, vmax = np.nanpercentile(finite, percentiles)
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 0.5, vmax + 0.5
        levels = np.linspace(vmin, vmax, int(level_count))
        return levels, plt.get_cmap(cmap_name), None

    levels = np.asarray(levels, dtype=float)
    if levels[0] < 0 < levels[-1]:
        cmap, norm = color.make_cmap(levels, neg_color=neg_color, pos_color=pos_color)
        return levels, cmap, norm

    cmap = ListedColormap(color.myColor(levels, min_color, max_color))
    norm = BoundaryNorm(levels, cmap.N)
    return levels, cmap, norm


def _set_axis_labels_and_ticks(ax, xlabel=None, ylabel=None, xlabel_fontsize=None, ylabel_fontsize=None, x_tick_count=None, y_tick_count=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if x_tick_count is not None and int(x_tick_count) >= 2:
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.linspace(xmin, xmax, int(x_tick_count)))
    if y_tick_count is not None and int(y_tick_count) >= 2:
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.linspace(ymin, ymax, int(y_tick_count)))


def _map_tick_values(vmin, vmax, tick_count, default_step=5):
    if tick_count is not None and int(tick_count) >= 2:
        return np.linspace(vmin, vmax, int(tick_count))
    return np.arange(np.floor(vmin), np.ceil(vmax) + 1, default_step)


def _finite_points(lats, lons):
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    valid = np.isfinite(lats) & np.isfinite(lons)
    return lats[valid], lons[valid]


def _expand_extent_with_points(lat_min, lat_max, lon_min, lon_max, lats, lons, margin_degrees=1.0):
    point_lats, point_lons = _finite_points(lats, lons)
    if point_lats.size == 0:
        return lat_min, lat_max, lon_min, lon_max

    margin = max(0.0, float(margin_degrees))
    lat_min = min(lat_min, float(np.nanmin(point_lats)) - margin)
    lat_max = max(lat_max, float(np.nanmax(point_lats)) + margin)
    lon_min = min(lon_min, float(np.nanmin(point_lons)) - margin)
    lon_max = max(lon_max, float(np.nanmax(point_lons)) + margin)

    # Mercator cannot include the poles.
    lat_min = max(lat_min, -89.0)
    lat_max = min(lat_max, 89.0)
    return lat_min, lat_max, lon_min, lon_max


def _count_points_in_extent(lats, lons, lat_min, lat_max, lon_min, lon_max):
    point_lats, point_lons = _finite_points(lats, lons)
    if point_lats.size == 0:
        return 0
    inside = (
        (point_lats >= lat_min)
        & (point_lats <= lat_max)
        & (point_lons >= lon_min)
        & (point_lons <= lon_max)
    )
    return int(np.count_nonzero(inside))


def _pressure_for_display(values, pressure_name):
    display = np.asarray(values, dtype=float)
    if np.nanmedian(np.abs(display)) > 2000:
        display = display / 100.0
    if pressure_name.upper() == "PSFC":
        label = "Surface pressure (PSFC fallback) [hPa]"
    else:
        label = f"Sea-level pressure ({pressure_name}) [hPa]"
    return display, label


def _choose_pressure_variable(dataset, pressure_var_candidates):
    for name in pressure_var_candidates:
        if name in dataset:
            return name
    available = ", ".join(list(dataset.data_vars)[:30])
    raise KeyError(
        f"No pressure variable was found. Tried {pressure_var_candidates}. "
        f"First available variables: {available}"
    )


def _pressure_plane(dataset, pressure_name, time_index=0):
    pressure = _select_time_if_present(dataset[pressure_name], time_index).squeeze(drop=True)
    if pressure.ndim != 2:
        raise ValueError(f"Expected a 2D pressure field after time selection; got dims {pressure.dims}")
    return pressure.values


def _sea_level_pressure_plane(dataset, time_index=0, pressure_var_candidates=("SLP", "slp", "PMSL", "MSLP", "PSFC")):
    for name in pressure_var_candidates:
        if name in dataset and name.upper() != "PSFC":
            pressure = np.asarray(_pressure_plane(dataset, name, time_index), dtype=float)
            if np.nanmedian(np.abs(pressure)) > 2000:
                pressure = pressure / 100.0
            return pressure, name

    if all(name in dataset for name in ("PSFC", "T2", "HGT")):
        psfc = np.asarray(_pressure_plane(dataset, "PSFC", time_index), dtype=float)
        psfc_pa = psfc if np.nanmedian(np.abs(psfc)) > 2000 else psfc * 100.0
        t2 = _select_time_if_present(dataset["T2"], time_index).squeeze(drop=True).values.astype(float)
        hgt = _select_time_if_present(dataset["HGT"], time_index).squeeze(drop=True).values.astype(float)
        if t2.shape != psfc_pa.shape or hgt.shape != psfc_pa.shape:
            raise ValueError(
                f"Cannot compute sea-level pressure: PSFC {psfc_pa.shape}, T2 {t2.shape}, HGT {hgt.shape}"
            )
        if "Q2" in dataset:
            q2 = _select_time_if_present(dataset["Q2"], time_index).squeeze(drop=True).values.astype(float)
            if q2.shape == t2.shape:
                t2 = t2 * (1.0 + 0.61 * q2)

        g = 9.80665
        rd = 287.05
        lapse_rate = 0.0065
        hgt_pos = np.maximum(hgt, 0.0)
        mean_temperature = np.maximum(t2 + 0.5 * lapse_rate * hgt_pos, 180.0)
        slp_hpa = psfc_pa * np.exp(g * hgt_pos / (rd * mean_temperature)) / 100.0
        return slp_hpa, "computed_SLP_from_PSFC_T2_HGT"

    if "PSFC" in dataset:
        warnings.warn(
            "No SLP/PMSL/MSLP variable and not enough fields to compute SLP; using raw PSFC for pressure tracking.",
            RuntimeWarning,
        )
        pressure = np.asarray(_pressure_plane(dataset, "PSFC", time_index), dtype=float)
        if np.nanmedian(np.abs(pressure)) > 2000:
            pressure = pressure / 100.0
        return pressure, "PSFC"

    pressure_name = _choose_pressure_variable(dataset, pressure_var_candidates)
    pressure = np.asarray(_pressure_plane(dataset, pressure_name, time_index), dtype=float)
    if np.nanmedian(np.abs(pressure)) > 2000:
        pressure = pressure / 100.0
    return pressure, pressure_name


def _land_ocean_masks(dataset, landmask_var_candidates, time_index=0):
    mask_name = next((name for name in landmask_var_candidates if name in dataset), None)
    if mask_name is None:
        raise KeyError(f"No land mask variable was found. Tried {landmask_var_candidates}.")

    raw = _select_time_if_present(dataset[mask_name], time_index).squeeze(drop=True).values.astype(float)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D land mask after time selection; got shape {raw.shape}")

    if mask_name.upper() == "XLAND":
        land = np.isfinite(raw) & (raw < 1.5)
        ocean = np.isfinite(raw) & (raw >= 1.5)
    else:
        land = np.isfinite(raw) & (raw >= 0.5)
        ocean = np.isfinite(raw) & (raw < 0.5)
    return land, ocean, mask_name


def _low_elevation_ocean_mask(dataset, ocean_mask, time_index=0, max_ocean_hgt_m=100.0):
    if max_ocean_hgt_m is None:
        return ocean_mask
    if "HGT" not in dataset:
        warnings.warn("HGT was not found; ocean pressure tracking will use the raw ocean mask.", RuntimeWarning)
        return ocean_mask
    hgt = _select_time_if_present(dataset["HGT"], time_index).squeeze(drop=True).values.astype(float)
    if hgt.shape != ocean_mask.shape:
        warnings.warn(
            f"HGT shape {hgt.shape} does not match ocean mask shape {ocean_mask.shape}; using raw ocean mask.",
            RuntimeWarning,
        )
        return ocean_mask
    return ocean_mask & np.isfinite(hgt) & (hgt <= float(max_ocean_hgt_m))


def _transpose_with_vertical_first(data_array, vertical_names):
    vertical_dim = next((dim for dim in data_array.dims if dim in vertical_names), None)
    if vertical_dim is None:
        raise ValueError(f"Could not find one of {vertical_names} in dims {data_array.dims}")
    horizontal_dims = [dim for dim in data_array.dims if dim != vertical_dim]
    if len(horizontal_dims) != 2:
        raise ValueError(f"Expected two horizontal dimensions in {data_array.dims}")
    return data_array.transpose(vertical_dim, *horizontal_dims)


def _interpolate_to_pressure_level(pressure_pa, field, target_pa):
    pressure = np.asarray(pressure_pa, dtype=float)
    values = np.asarray(field, dtype=float)
    if pressure.shape != values.shape:
        raise ValueError(f"Pressure and field shapes differ: {pressure.shape} vs {values.shape}")

    if np.nanmean(pressure[0]) < np.nanmean(pressure[-1]):
        pressure = pressure[::-1]
        values = values[::-1]

    crosses = (pressure[:-1] >= target_pa) & (pressure[1:] <= target_pa)
    has_crossing = crosses.any(axis=0)
    k = np.argmax(crosses, axis=0)
    yy, xx = np.indices(has_crossing.shape)

    p1 = pressure[k, yy, xx]
    p2 = pressure[k + 1, yy, xx]
    v1 = values[k, yy, xx]
    v2 = values[k + 1, yy, xx]

    out = np.full(has_crossing.shape, np.nan, dtype=float)
    denom = p2 - p1
    ok = has_crossing & np.isfinite(p1) & np.isfinite(p2) & np.isfinite(v1) & np.isfinite(v2) & (denom != 0)
    weight = (target_pa - p1[ok]) / denom[ok]
    out[ok] = v1[ok] + weight * (v2[ok] - v1[ok])
    return out


def _geopotential_height_perturbation_at_pressure(dataset, target_hpa=850.0, time_index=0):
    required = ("P", "PB", "PH")
    missing = [name for name in required if name not in dataset]
    if missing:
        raise KeyError(
            f"Cannot compute {target_hpa:g}-hPa geopotential height perturbation from PH; missing {missing}"
        )

    p_da = _transpose_with_vertical_first(
        _select_time_if_present(dataset["P"], time_index).squeeze(drop=True),
        ("bottom_top", "level", "lev"),
    )
    pb_da = _transpose_with_vertical_first(
        _select_time_if_present(dataset["PB"], time_index).squeeze(drop=True),
        ("bottom_top", "level", "lev"),
    )
    ph_da = _transpose_with_vertical_first(
        _select_time_if_present(dataset["PH"], time_index).squeeze(drop=True),
        ("bottom_top_stag", "level_stag", "lev_stag"),
    )

    pressure_pa = p_da.values + pb_da.values
    height_perturbation_mass = 0.5 * (ph_da.values[:-1] + ph_da.values[1:]) / 9.80665
    return _interpolate_to_pressure_level(pressure_pa, height_perturbation_mass, target_hpa * 100.0)


def _wind10m_speed(dataset, time_index=0):
    if "U10" not in dataset or "V10" not in dataset:
        raise KeyError("Maximum 10-m wind criterion requires U10 and V10 variables.")
    u10 = _select_time_if_present(dataset["U10"], time_index).squeeze(drop=True).values
    v10 = _select_time_if_present(dataset["V10"], time_index).squeeze(drop=True).values
    return np.sqrt(np.square(u10) + np.square(v10))


def _center_dict(row, col, lats, lons, field, method, units):
    return {
        "method": method,
        "row": int(row),
        "col": int(col),
        "latitude": float(lats[row, col]),
        "longitude": float(lons[row, col]),
        "value": float(field[row, col]),
        "units": units,
    }


def _masked_min_center(field, valid_mask, lats, lons, method, units):
    valid = valid_mask & np.isfinite(field)
    if not valid.any():
        raise ValueError(f"No valid points for {method}")
    search = np.where(valid, field, np.inf)
    row, col = _wrfplot().min_pressure_center_index(search)
    return _center_dict(row, col, lats, lons, field, method, units)


def _masked_max_center(field, valid_mask, lats, lons, method, units):
    valid = valid_mask & np.isfinite(field)
    if not valid.any():
        raise ValueError(f"No valid points for {method}")
    search = np.where(valid, field, -np.inf)
    row, col = np.unravel_index(np.nanargmax(search), search.shape)
    return _center_dict(row, col, lats, lons, field, method, units)


def _haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _pairwise_distances_km(centers):
    names = list(centers)
    distances = {}
    for i, name1 in enumerate(names):
        for name2 in names[i + 1:]:
            c1 = centers[name1]
            c2 = centers[name2]
            distances[f"{name1}_to_{name2}_km"] = _haversine_km(
                c1["latitude"], c1["longitude"], c2["latitude"], c2["longitude"]
            )
    return distances


def _apply_center_continuity(
    track_rows,
    max_center_jump_km_per_6h=50.0,
    center_jump_reference_hours=6.0,
    filename_time_regex=DEFAULT_TIME_REGEX,
    filename_time_format=DEFAULT_TIME_FORMAT,
):
    if max_center_jump_km_per_6h is None:
        return track_rows

    max_jump = float(max_center_jump_km_per_6h)
    reference_hours = float(center_jump_reference_hours)
    if reference_hours <= 0:
        raise ValueError("center_jump_reference_hours must be positive.")

    previous = None
    previous_time = None
    for row in track_rows:
        row.update(
            {
                "candidate_latitude": row["latitude"],
                "candidate_longitude": row["longitude"],
                "candidate_row": row["row"],
                "candidate_col": row["col"],
                "candidate_selected_method": row["selected_method"],
                "candidate_selection_rule": row["selection_rule"],
            }
        )

        current_time, _ = _parse_filename_time(row["file"], filename_time_regex, filename_time_format)
        if previous is None:
            row.update(
                {
                    "continuity_adjusted": False,
                    "continuity_distance_km": np.nan,
                    "continuity_allowed_distance_km": np.nan,
                    "continuity_hours_since_previous": np.nan,
                    "continuity_rule": "accepted_first_center",
                }
            )
            previous = row
            previous_time = current_time
            continue

        hours_since_previous = reference_hours
        if previous_time is not None and current_time is not None:
            hours_since_previous = abs((current_time - previous_time).total_seconds()) / 3600.0
            if not np.isfinite(hours_since_previous) or hours_since_previous <= 0:
                hours_since_previous = reference_hours

        allowed_distance = max_jump * hours_since_previous / reference_hours
        distance = _haversine_km(
            previous["latitude"],
            previous["longitude"],
            row["candidate_latitude"],
            row["candidate_longitude"],
        )

        row.update(
            {
                "continuity_distance_km": distance,
                "continuity_allowed_distance_km": allowed_distance,
                "continuity_hours_since_previous": hours_since_previous,
            }
        )
        if distance > allowed_distance:
            row.update(
                {
                    "latitude": previous["latitude"],
                    "longitude": previous["longitude"],
                    "row": previous["row"],
                    "col": previous["col"],
                    "selected_method": "previous accepted center",
                    "selection_rule": "held_previous_center_jump_exceeded",
                    "continuity_adjusted": True,
                    "continuity_rule": "held_previous_center_jump_exceeded",
                }
            )
        else:
            row.update(
                {
                    "continuity_adjusted": False,
                    "continuity_rule": "accepted_within_jump_limit",
                }
            )

        previous = row
        previous_time = current_time if current_time is not None else previous_time

    return track_rows


def _parse_atcf_coordinate(text):
    text = text.strip().upper()
    if not text:
        return np.nan
    hemisphere = text[-1]
    value = float(text[:-1]) / 10.0
    if hemisphere in ("S", "W"):
        value = -value
    return value


def parse_observed_atcf_track(path, output_time_format="%Y-%m-%d_%H:%M:%S"):
    """Read an ATCF/JTWC best-track file into a list of track dictionaries."""
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                continue
            if len(parts) > 4 and parts[4].upper() != "BEST":
                continue
            try:
                valid_time = datetime.strptime(parts[2], "%Y%m%d%H")
                lat = _parse_atcf_coordinate(parts[6])
                lon = _parse_atcf_coordinate(parts[7])
            except (ValueError, IndexError):
                continue
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            rows.append(
                {
                    "time": valid_time.strftime(output_time_format),
                    "datetime": valid_time,
                    "latitude": lat,
                    "longitude": lon,
                    "vmax_kt": float(parts[8]) if len(parts) > 8 and parts[8] else np.nan,
                    "pressure_hpa": float(parts[9]) if len(parts) > 9 and parts[9] else np.nan,
                }
            )
    rows.sort(key=lambda row: row["datetime"])
    return rows

def _find_vortex_center(
    path,
    time_index=0,
    target_pressure_hpa=850.0,
    center_agreement_km=100.0,
    max_ocean_hgt_m=100.0,
    pressure_var_candidates=("SLP", "slp", "PMSL", "MSLP", "PSFC"),
    landmask_var_candidates=("LANDMASK", "XLAND"),
    filename_time_regex=DEFAULT_TIME_REGEX,
    filename_time_format=DEFAULT_TIME_FORMAT,
):
    errors = []
    with _xr().open_dataset(path) as dataset:
        pressure, pressure_name = _sea_level_pressure_plane(dataset, time_index, pressure_var_candidates)
        lats, lons = _lat_lon_arrays(dataset, time_index)
        _land_mask, ocean_mask, landmask_name = _land_ocean_masks(dataset, landmask_var_candidates, time_index)
        ocean_pressure_mask = _low_elevation_ocean_mask(dataset, ocean_mask, time_index, max_ocean_hgt_m)
        z850_perturbation = _geopotential_height_perturbation_at_pressure(dataset, target_pressure_hpa, time_index)
        speed10 = _wind10m_speed(dataset, time_index)

        try:
            pressure_ocean = _masked_min_center(
                pressure,
                ocean_pressure_mask,
                lats,
                lons,
                f"minimum {pressure_name} over ocean",
                pressure_name,
            )
        except Exception as exc:
            pressure_ocean = None
            errors.append(f"pressure_ocean: {exc}")

        z850_perturbation_center = _masked_min_center(
            z850_perturbation,
            np.ones_like(z850_perturbation, dtype=bool),
            lats,
            lons,
            f"minimum {target_pressure_hpa:g}-hPa geopotential height perturbation",
            "m",
        )

        try:
            wind10m = _masked_max_center(
                speed10,
                np.ones_like(speed10, dtype=bool),
                lats,
                lons,
                "maximum 10-m wind",
                "m s-1",
            )
        except Exception as exc:
            wind10m = None
            errors.append(f"wind10m: {exc}")

        criteria = {
            "pressure_ocean": pressure_ocean,
            "z850_perturbation": z850_perturbation_center,
            "wind10m": wind10m,
        }
        complete = {name: center for name, center in criteria.items() if center is not None}
        distances = _pairwise_distances_km(complete) if len(complete) > 1 else {}
        max_distance = max(distances.values()) if distances else np.nan

        final_center = pressure_ocean or z850_perturbation_center
        if pressure_ocean is None:
            selection_rule = "fallback_z850_perturbation_missing_pressure_ocean"
        elif len(complete) == 3 and max_distance <= center_agreement_km:
            selection_rule = "criteria_agree_use_pressure_ocean"
        elif len(complete) == 3:
            selection_rule = "fallback_pressure_ocean_disagreement"
        else:
            selection_rule = "fallback_pressure_ocean_missing_criterion"

        parsed_time, text = _parse_filename_time(path, filename_time_regex, filename_time_format)
        time_label = parsed_time.strftime(filename_time_format) if parsed_time is not None else text
        row = {
            "time": time_label,
            "latitude": final_center["latitude"],
            "longitude": final_center["longitude"],
            "row": final_center["row"],
            "col": final_center["col"],
            "selected_method": final_center["method"],
            "selection_rule": selection_rule,
            "max_criteria_distance_km": max_distance,
            "pressure_name": pressure_name,
            "landmask_name": landmask_name,
            "max_ocean_hgt_m": max_ocean_hgt_m,
            "file": Path(path).name,
            "errors": "; ".join(errors),
        }

        for prefix, center in criteria.items():
            if center is None:
                row.update(
                    {
                        f"{prefix}_lat": np.nan,
                        f"{prefix}_lon": np.nan,
                        f"{prefix}_row": np.nan,
                        f"{prefix}_col": np.nan,
                        f"{prefix}_value": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        f"{prefix}_lat": center["latitude"],
                        f"{prefix}_lon": center["longitude"],
                        f"{prefix}_row": center["row"],
                        f"{prefix}_col": center["col"],
                        f"{prefix}_value": center["value"],
                    }
                )
        row.update(distances)
        return row


def _map_field_2d(
    dataset,
    varname,
    time_index=0,
    level_index=0,
    pressure_var_candidates=("SLP", "slp", "PMSL", "MSLP", "PSFC"),
):
    lower = str(varname).lower()
    if lower in ("pressure", "slp", "sea_level_pressure", "sea-level pressure"):
        values, pressure_name = _sea_level_pressure_plane(dataset, time_index, pressure_var_candidates)
        if pressure_name == "computed_SLP_from_PSFC_T2_HGT":
            label = "Sea-level pressure (computed from PSFC, T2, HGT) [hPa]"
        elif pressure_name.upper() == "PSFC":
            label = "Surface pressure (PSFC fallback) [hPa]"
        else:
            label = f"Sea-level pressure ({pressure_name}) [hPa]"
        return values, label

    if lower in ("wind10", "wind10m", "10m_wind", "10m_wind_speed"):
        return _wind10m_speed(dataset, time_index), "10-m wind speed [m s-1]"

    if varname not in dataset:
        raise KeyError(f"Shading/vector variable {varname!r} was not found in the dataset.")

    data_array = _select_time_if_present(dataset[varname], time_index).squeeze(drop=True)
    ydim, xdim = _guess_xy_dims(data_array)
    level_dim = _guess_level_dim(data_array, None, ydim, xdim)
    if level_dim is not None and data_array.ndim > 2:
        if not 0 <= int(level_index) < data_array.sizes[level_dim]:
            raise IndexError(f"level_index={level_index} is outside 0:{data_array.sizes[level_dim] - 1}")
        data_array = data_array.isel({level_dim: int(level_index)}).squeeze(drop=True)

    if data_array.ndim != 2:
        raise ValueError(f"Variable {varname!r} must be 2D after time/level selection; got dims {data_array.dims}")

    ydim, xdim = _guess_xy_dims(data_array)
    data_array = data_array.transpose(ydim, xdim)
    long_name = data_array.attrs.get("description") or data_array.attrs.get("long_name") or varname
    units = data_array.attrs.get("units", "")
    label = f"{long_name} [{units}]" if units else str(long_name)
    return np.asarray(data_array.values, dtype=float), label


def _height_mass_km(dataset, time_index=0):
    if "PH" not in dataset or "PHB" not in dataset:
        raise KeyError("PH and PHB are required to compute geometric height.")
    ph = _select_time_if_present(dataset["PH"], time_index).squeeze(drop=True).values
    phb = _select_time_if_present(dataset["PHB"], time_index).squeeze(drop=True).values
    height_stag_km = (ph + phb) / 9.80665 / 1000.0
    return 0.5 * (height_stag_km[:-1, :, :] + height_stag_km[1:, :, :])


def _pressure_hpa_3d(dataset, time_index=0):
    if "P" not in dataset or "PB" not in dataset:
        raise KeyError("P and PB are required to compute pressure_hpa.")
    p = _select_time_if_present(dataset["P"], time_index).squeeze(drop=True).values
    pb = _select_time_if_present(dataset["PB"], time_index).squeeze(drop=True).values
    return (p + pb) / 100.0


def _mass_grid_dimensions(dataset):
    if "T" in dataset and {"bottom_top", "south_north", "west_east"}.issubset(dataset["T"].dims):
        return dataset.sizes["west_east"], dataset.sizes["south_north"], dataset.sizes["bottom_top"]
    nx = dataset.sizes.get("west_east")
    ny = dataset.sizes.get("south_north")
    nz = dataset.sizes.get("bottom_top")
    if nx is None or ny is None or nz is None:
        raise ValueError("Could not infer WRF mass-grid dimensions nx, ny, nz.")
    return int(nx), int(ny), int(nz)


def _unstagger_to_mass_grid(dataset, varname, time_index=0):
    nx, ny, _ = _mass_grid_dimensions(dataset)
    if varname == "U":
        u = _select_time_if_present(dataset["U"], time_index).squeeze(drop=True).values
        return 0.5 * (u[:, :, 0:nx] + u[:, :, 1 : nx + 1])
    if varname == "V":
        v = _select_time_if_present(dataset["V"], time_index).squeeze(drop=True).values
        return 0.5 * (v[:, 0:ny, :] + v[:, 1 : ny + 1, :])
    if varname == "W":
        w = _select_time_if_present(dataset["W"], time_index).squeeze(drop=True).values
        return 0.5 * (w[:-1, :, :] + w[1:, :, :])
    raise ValueError(f"{varname} is not a known staggered 3D wind variable")


def _variable_3d(dataset, varname, time_index=0, variable_scale=None, variable_units_override=None):
    variable_scale = variable_scale or {}
    variable_units_override = variable_units_override or {}
    lower = varname.lower()

    if lower == "pressure_hpa":
        values = _pressure_hpa_3d(dataset, time_index)
        units = variable_units_override.get("pressure_hpa", "hPa")
    elif lower == "height_km":
        values = _height_mass_km(dataset, time_index)
        units = variable_units_override.get("height_km", "km")
    elif lower == "wind_speed":
        u = _unstagger_to_mass_grid(dataset, "U", time_index)
        v = _unstagger_to_mass_grid(dataset, "V", time_index)
        values = np.sqrt(u * u + v * v)
        units = variable_units_override.get("wind_speed", "m s-1")
    elif varname in ("U", "V", "W"):
        values = _unstagger_to_mass_grid(dataset, varname, time_index)
        units = dataset[varname].attrs.get("units", "") if varname in dataset else ""
    else:
        if varname not in dataset:
            raise KeyError(f"Variable {varname!r} was not found in the dataset.")
        da = _select_time_if_present(dataset[varname], time_index).squeeze(drop=True)
        if da.ndim != 3:
            raise ValueError(f"Variable {varname!r} must be 3D after time selection; got dims {da.dims}")
        if {"bottom_top", "south_north", "west_east"}.issubset(da.dims):
            da = da.transpose("bottom_top", "south_north", "west_east")
        values = da.values
        units = dataset[varname].attrs.get("units", "")

    scale = variable_scale.get(varname, variable_scale.get(lower, 1.0))
    values = np.asarray(values, dtype=float) * scale
    units = variable_units_override.get(varname, variable_units_override.get(lower, units))
    return values, units


def _nearest_section_index(axis, value, lats, lons):
    axis = axis.lower()
    if axis == "lat":
        lat1d = lats[:, 0]
        return int(np.abs(value - lat1d).argmin())
    if axis == "lon":
        lon1d = lons[0, :]
        return int(np.abs(value - lon1d).argmin())
    raise ValueError('cross_section_axis must be either "lat" or "lon"')


def _cumulative_distance_km(section_lats, section_lons):
    dist = np.zeros(section_lats.size, dtype=float)
    for i in range(1, section_lats.size):
        dist[i] = dist[i - 1] + _haversine_km(section_lats[i - 1], section_lons[i - 1], section_lats[i], section_lons[i])
    return dist


def _extract_section(field3d, axis, index):
    if axis.lower() == "lat":
        return field3d[:, index, :]
    return field3d[:, :, index]


def _make_signed_levels(values, level_count, symmetric=True):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot build levels because the selected cross section has no finite values.")
    level_count = max(3, int(level_count))
    if level_count % 2 == 0:
        level_count += 1
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0.0) else 0.05 * abs(vmin)
        vmin, vmax = vmin - delta, vmax + delta
    if symmetric and vmin < 0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return np.linspace(-bound, bound, level_count)
    return np.linspace(vmin, vmax, level_count)


def _signed_shading_cmap(
    levels,
    negative_shade_color=(0.0, 0.15, 0.85),
    positive_shade_color=(0.85, 0.0, 0.0),
    neutral_color=(1.0, 1.0, 1.0),
):
    levels = np.asarray(levels, dtype=float)
    midpoints = 0.5 * (levels[:-1] + levels[1:])
    nneg = int(np.sum(midpoints < 0))
    nzero = int(np.sum(np.isclose(midpoints, 0.0)))
    npos = int(np.sum(midpoints > 0))

    colors = []
    if nneg:
        colors.extend(color.myColor(np.arange(nneg + 1), negative_shade_color, neutral_color))
    colors.extend([neutral_color] * nzero)
    if npos:
        colors.extend(color.myColor(np.arange(npos + 1), neutral_color, positive_shade_color))
    if len(colors) != len(levels) - 1:
        colors = color.myColor(levels, neutral_color, positive_shade_color)
    return ListedColormap(colors), BoundaryNorm(levels, len(colors))


def _draw_signed_contours(
    ax,
    xgrid,
    zgrid,
    values,
    levels,
    negative_color="blue",
    positive_color="red",
    zero_color="black",
    negative_linestyle="dotted",
    positive_linestyle="solid",
    linewidth=1.1,
    label_fontsize=8,
):
    levels = np.asarray(levels, dtype=float)
    neg_levels = levels[levels < 0]
    pos_levels = levels[levels > 0]
    artists = []
    if neg_levels.size:
        cs_neg = ax.contour(
            xgrid,
            zgrid,
            values,
            levels=neg_levels,
            colors=negative_color,
            linestyles=negative_linestyle,
            linewidths=linewidth,
        )
        ax.clabel(cs_neg, inline=True, fontsize=label_fontsize, fmt="%g")
        artists.append(cs_neg)
    if np.nanmin(values) <= 0 <= np.nanmax(values):
        cs_zero = ax.contour(
            xgrid,
            zgrid,
            values,
            levels=[0],
            colors=zero_color,
            linestyles="solid",
            linewidths=linewidth,
        )
        ax.clabel(cs_zero, inline=True, fontsize=label_fontsize, fmt="%g")
        artists.append(cs_zero)
    if pos_levels.size:
        cs_pos = ax.contour(
            xgrid,
            zgrid,
            values,
            levels=pos_levels,
            colors=positive_color,
            linestyles=positive_linestyle,
            linewidths=linewidth,
        )
        ax.clabel(cs_pos, inline=True, fontsize=label_fontsize, fmt="%g")
        artists.append(cs_pos)
    return artists


def plot_animation_d03(
    files,
    varname,
    level_index=0,
    time_index=0,
    latlon_bounds=None,
    index_bounds=None,
    levels=None,
    level_count=21,
    cmap_name="viridis",
    min_color=(0.9, 0.9, 1.0),
    max_color=(0.0, 0.0, 1.0),
    neg_color=(0.0, 0.0, 1.0),
    pos_color=(1.0, 0.0, 0.0),
    percentiles=(2, 98),
    interval_ms=300,
    figsize=(8, 6),
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    x_tick_count=None,
    y_tick_count=None,
    repeat=True,
    cache_frames=True,
    filename_time_regex=DEFAULT_TIME_REGEX,
    filename_time_format=DEFAULT_TIME_FORMAT,
):
    """
    Animate one shaded variable over a list of NetCDF/WRF files.

    Returns
    -------
    fig, animation
        Keep the returned animation object alive in notebooks, for example:
        ``fig, ani = plot_animation_d03(...); HTML(ani.to_jshtml())``.
    """
    frame_files = _sort_files(files, filename_time_regex, filename_time_format)
    if not frame_files:
        raise ValueError("files is empty")

    with _xr().open_dataset(frame_files[0]) as ds0:
        if varname not in ds0:
            raise KeyError(f"Variable {varname!r} was not found in {frame_files[0]}")
        da = ds0[varname]
        time_dim = _guess_time_dim(da)
        ydim, xdim = _guess_xy_dims(da)
        level_dim = _guess_level_dim(da, time_dim, ydim, xdim)

        if level_dim is not None and not 0 <= level_index < da.sizes[level_dim]:
            raise IndexError(f"level_index={level_index} is outside 0:{da.sizes[level_dim] - 1}")

        y_size = da.sizes[ydim]
        x_size = da.sizes[xdim]
        if latlon_bounds is not None:
            rs, re, cs, ce = _indices_from_latlon(ds0, *latlon_bounds)
            rs, re = _normalize_bounds(rs, re, y_size)
            cs, ce = _normalize_bounds(cs, ce, x_size)
        elif index_bounds is not None:
            rs, re, cs, ce = index_bounds
            rs, re = _normalize_bounds(rs, re, y_size)
            cs, ce = _normalize_bounds(cs, ce, x_size)
        else:
            rs, re = 0, y_size
            cs, ce = 0, x_size

        row_slice = slice(rs, re)
        col_slice = slice(cs, ce)

        if "XLAT" in ds0 and "XLONG" in ds0:
            lat = _drop_time_like(ds0["XLAT"])
            lon = _drop_time_like(ds0["XLONG"])
            if lat.ndim == 2 and lon.ndim == 2:
                xgrid = lon.values[row_slice, col_slice]
                ygrid = lat.values[row_slice, col_slice]
                xlabel, ylabel = "Longitude", "Latitude"
            else:
                xvals = np.arange(cs, ce)
                yvals = np.arange(rs, re)
                xgrid, ygrid = np.meshgrid(xvals, yvals)
                xlabel, ylabel = xdim, ydim
        else:
            dx_raw = getattr(ds0, "DX", 1.0)
            dx_raw = float(dx_raw) if np.isscalar(dx_raw) else float(np.asarray(dx_raw).ravel()[0])
            dx_km = dx_raw / 1000.0 if dx_raw > 100.0 else dx_raw
            xvals = (np.arange(cs, ce) - 0.5 * (x_size - 1)) * dx_km
            yvals = (np.arange(rs, re) - 0.5 * (y_size - 1)) * dx_km
            xgrid, ygrid = np.meshgrid(xvals, yvals)
            xlabel, ylabel = "X (km)", "Y (km)"

        long_name = da.attrs.get("description") or da.attrs.get("long_name") or varname
        units = da.attrs.get("units", "")
        cbar_label = f"{long_name} ({units})" if units else long_name
        lev_label = "" if level_dim is None else f", {level_dim} index={level_index}"

    frame_cache = {}

    def select_frame_values(dataset):
        frame = dataset[varname]
        if time_dim is not None and time_dim in frame.dims:
            frame = frame.isel({time_dim: time_index})
        if level_dim is not None and level_dim in frame.dims:
            frame = frame.isel({level_dim: level_index})
        frame = frame.squeeze(drop=True)
        frame = frame.transpose(ydim, xdim)
        return np.asarray(frame.isel({ydim: row_slice, xdim: col_slice}).values).copy()

    def get_frame(frame_index):
        if cache_frames and frame_index in frame_cache:
            return frame_cache[frame_index]
        with _xr().open_dataset(frame_files[frame_index]) as frame_ds:
            values = select_frame_values(frame_ds)
        if cache_frames:
            frame_cache[frame_index] = values
        return values

    if levels is None:
        stride = max(1, len(frame_files) // 10)
        samples = []
        for frame_index in range(0, len(frame_files), stride):
            values = np.asarray(get_frame(frame_index), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                samples.append(finite)
        if not samples:
            raise ValueError("No finite data values were found in the selected plot domain.")
        sample_values = np.concatenate(samples)
    else:
        sample_values = get_frame(0)

    levels, cmap, norm = _levels_and_cmap(
        sample_values,
        levels=levels,
        level_count=level_count,
        cmap_name=cmap_name,
        min_color=min_color,
        max_color=max_color,
        neg_color=neg_color,
        pos_color=pos_color,
        percentiles=percentiles,
    )

    fig, ax = plt.subplots(figsize=figsize)
    first_plot = ax.contourf(xgrid, ygrid, get_frame(0), levels=levels, cmap=cmap, norm=norm, extend="both")
    cbar = fig.colorbar(first_plot, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_xlim(np.nanmin(xgrid), np.nanmax(xgrid))
    ax.set_ylim(np.nanmin(ygrid), np.nanmax(ygrid))
    _set_axis_labels_and_ticks(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        x_tick_count=x_tick_count,
        y_tick_count=y_tick_count,
    )

    frame_labels = [_file_time_label(path, filename_time_regex, filename_time_format) for path in frame_files]

    def draw(frame_index):
        ax.clear()
        ax.contourf(xgrid, ygrid, get_frame(frame_index), levels=levels, cmap=cmap, norm=norm, extend="both")
        ax.set_title(f"{varname}{lev_label}: {frame_labels[frame_index]}")
        ax.set_xlim(np.nanmin(xgrid), np.nanmax(xgrid))
        ax.set_ylim(np.nanmin(ygrid), np.nanmax(ygrid))
        _set_axis_labels_and_ticks(
            ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            x_tick_count=x_tick_count,
            y_tick_count=y_tick_count,
        )
        return []

    animation = FuncAnimation(fig, draw, frames=len(frame_files), interval=interval_ms, blit=False, repeat=repeat)
    return fig, animation

def plot_track(
    files,
    observed_track_file=None,
    shading_var="pressure",
    overlay_vectors=True,
    time_index=0,
    shading_level_index=0,
    vector_u_var="U10",
    vector_v_var="V10",
    vector_level_index=0,
    target_pressure_hpa=850.0,
    center_agreement_km=100.0,
    max_ocean_hgt_m=100.0,
    max_center_jump_km_per_6h=50.0,
    center_jump_reference_hours=6.0,
    pressure_var_candidates=("SLP", "slp", "PMSL", "MSLP", "PSFC"),
    landmask_var_candidates=("LANDMASK", "XLAND"),
    shading_levels=None,
    shading_level_count=24,
    shading_percentiles=(1, 99),
    shading_cmap_name="viridis",
    shading_min_color=(0.95, 0.95, 1.0),
    shading_max_color=(0.0, 0.15, 0.75),
    shading_neg_color=(0.0, 0.0, 1.0),
    shading_pos_color=(1.0, 0.0, 0.0),
    model_track_color="red",
    model_track_marker_face="red",
    model_track_marker_edge="white",
    model_track_linewidth=4.0,
    model_track_marker_size=38,
    observed_track_color="black",
    observed_track_marker_face="yellow",
    observed_track_marker_edge=None,
    observed_track_linewidth=2.6,
    observed_track_marker_size=34,
    observed_track_linestyle="--",
    vector_skip=14,
    vector_scale=700,
    vector_width=0.0022,
    vector_alpha=0.75,
    vector_color="black",
    vector_key=20,
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    xtick_fontsize=None,
    ytick_fontsize=None,
    x_tick_count=None,
    y_tick_count=None,
    map_resolution="l",
    figsize=(10, 8),
    include_track_in_map_extent=True,
    map_margin_degrees=1.0,
    track_slice=None,
    filename_time_regex=DEFAULT_TIME_REGEX,
    filename_time_format=DEFAULT_TIME_FORMAT,
    track_output_file=None,
    diagnostics_output_file=None,
    save_path=None,
    return_data=False,
):
    """
    Track and plot Typhoon from WRF files and an observed ATCF best-track file.

    Tracking criteria:
    1. minimum SLP-like pressure over low-elevation ocean points, computing SLP
       from PSFC/T2/HGT if SLP is absent,
    2. minimum 850-hPa geopotential height perturbation from WRF PH,
    3. maximum 10-m wind speed,
    4. accepted centers cannot jump more than ``max_center_jump_km_per_6h`` every
       ``center_jump_reference_hours``.

    The main CSV track output, when requested, is written as ``time,lat,lon``.
    """
    wrf_files = _sort_files(files, filename_time_regex, filename_time_format)
    if not wrf_files:
        raise ValueError("files is empty")

    track_rows = [
        _find_vortex_center(
            path,
            time_index=time_index,
            target_pressure_hpa=target_pressure_hpa,
            center_agreement_km=center_agreement_km,
            max_ocean_hgt_m=max_ocean_hgt_m,
            pressure_var_candidates=pressure_var_candidates,
            landmask_var_candidates=landmask_var_candidates,
            filename_time_regex=filename_time_regex,
            filename_time_format=filename_time_format,
        )
        for path in wrf_files
    ]
    '''
    track_rows = _apply_center_continuity(
        track_rows,
        max_center_jump_km_per_6h=max_center_jump_km_per_6h,
        center_jump_reference_hours=center_jump_reference_hours,
        filename_time_regex=filename_time_regex,
        filename_time_format=filename_time_format,
    )
    '''
    track_rows_for_plot = track_rows[track_slice] if track_slice is not None else track_rows
    track_lats = np.array([row["latitude"] for row in track_rows_for_plot], dtype=float)
    track_lons = np.array([row["longitude"] for row in track_rows_for_plot], dtype=float)
    if track_lats.size == 0:
        warnings.warn("No model track rows are available for plotting after track_slice.", RuntimeWarning)

    observed_rows = []
    if observed_track_file is not None:
        observed_rows = parse_observed_atcf_track(observed_track_file, output_time_format=filename_time_format)
        if not observed_rows:
            warnings.warn(f"No observed track rows were parsed from {observed_track_file}.", RuntimeWarning)
    obs_lats = np.array([row["latitude"] for row in observed_rows], dtype=float)
    obs_lons = np.array([row["longitude"] for row in observed_rows], dtype=float)

    first_file = wrf_files[0]
    with _xr().open_dataset(first_file) as ds0:
        lats, lons = _lat_lon_arrays(ds0, time_index)
        shading_plot, shading_label = _map_field_2d(
            ds0,
            shading_var,
            time_index=time_index,
            level_index=shading_level_index,
            pressure_var_candidates=pressure_var_candidates,
        )
        if shading_plot.shape != lats.shape:
            raise ValueError(
                f"Shading variable {shading_var!r} has shape {shading_plot.shape}, "
                f"but XLAT/XLONG have shape {lats.shape}."
            )

        u_vector = None
        v_vector = None
        if overlay_vectors:
            if vector_u_var in ds0 and vector_v_var in ds0:
                u_vector, _ = _map_field_2d(
                    ds0,
                    vector_u_var,
                    time_index=time_index,
                    level_index=vector_level_index,
                    pressure_var_candidates=pressure_var_candidates,
                )
                v_vector, _ = _map_field_2d(
                    ds0,
                    vector_v_var,
                    time_index=time_index,
                    level_index=vector_level_index,
                    pressure_var_candidates=pressure_var_candidates,
                )
                if u_vector.shape != lats.shape or v_vector.shape != lats.shape:
                    warnings.warn(
                        f"Vector variables {vector_u_var!r}/{vector_v_var!r} do not match the map grid; "
                        "vector overlay will be skipped.",
                        RuntimeWarning,
                    )
                    u_vector = None
                    v_vector = None
            else:
                warnings.warn(
                    f"Vector variables {vector_u_var!r}/{vector_v_var!r} were not found; vector overlay skipped.",
                    RuntimeWarning,
                )

    levels, cmap, norm = _levels_and_cmap(
        shading_plot,
        levels=shading_levels,
        level_count=shading_level_count,
        cmap_name=shading_cmap_name,
        min_color=shading_min_color,
        max_color=shading_max_color,
        neg_color=shading_neg_color,
        pos_color=shading_pos_color,
        percentiles=shading_percentiles,
    )

    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    if include_track_in_map_extent:
        lat_min, lat_max, lon_min, lon_max = _expand_extent_with_points(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            track_lats,
            track_lons,
            margin_degrees=map_margin_degrees,
        )
        lat_min, lat_max, lon_min, lon_max = _expand_extent_with_points(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            obs_lats,
            obs_lons,
            margin_degrees=map_margin_degrees,
        )

    if track_lats.size and _count_points_in_extent(track_lats, track_lons, lat_min, lat_max, lon_min, lon_max) == 0:
        warnings.warn("No model track points fall inside the map extent.", RuntimeWarning)
    if obs_lats.size and _count_points_in_extent(obs_lats, obs_lons, lat_min, lat_max, lon_min, lon_max) == 0:
        warnings.warn("No observed track points fall inside the map extent.", RuntimeWarning)

    fig, ax = plt.subplots(figsize=figsize)
    m = Basemap(
        projection="merc",
        llcrnrlat=lat_min,
        urcrnrlat=lat_max,
        llcrnrlon=lon_min,
        urcrnrlon=lon_max,
        resolution=map_resolution,
        ax=ax,
    )

    x, y = m(lons, lats)
    shaded = m.pcolormesh(x, y, shading_plot, cmap=cmap, norm=norm, shading="auto")
    cbar = m.colorbar(shaded, location="bottom", pad="6%")
    cbar.set_label(shading_label)

    if overlay_vectors and u_vector is not None and v_vector is not None:
        skip = max(1, int(vector_skip))
        vector_slice = (slice(None, None, skip), slice(None, None, skip))
        q = m.quiver(
            x[vector_slice],
            y[vector_slice],
            u_vector[vector_slice],
            v_vector[vector_slice],
            scale=vector_scale,
            width=vector_width,
            color=vector_color,
            alpha=vector_alpha,
            zorder=4,
        )
        ax.quiverkey(
            q,
            0.88,
            1.03,
            vector_key,
            f"{vector_key:g} m/s",
            labelpos="E",
            coordinates="axes",
        )

    m.drawcoastlines(linewidth=0.7)
    m.drawcountries(linewidth=0.5)
    parallels = _map_tick_values(lat_min, lat_max, y_tick_count, default_step=5)
    meridians = _map_tick_values(lon_min, lon_max, x_tick_count, default_step=5)
    parallel_fontsize = ytick_fontsize if ytick_fontsize is not None else ylabel_fontsize
    meridian_fontsize = xtick_fontsize if xtick_fontsize is not None else xlabel_fontsize
    parallel_kwargs = {"fontsize": parallel_fontsize} if parallel_fontsize is not None else {}
    meridian_kwargs = {"fontsize": meridian_fontsize} if meridian_fontsize is not None else {}
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, **parallel_kwargs)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, **meridian_kwargs)
    ax.set_xlabel("Longitude", fontsize=xlabel_fontsize)
    ax.set_ylabel("Latitude", fontsize=ylabel_fontsize)

    if track_lats.size:
        tx, ty = m(track_lons, track_lats)
        m.plot(
            tx,
            ty,
            color=model_track_color,
            linewidth=model_track_linewidth,
            zorder=6,
            label="WRF center",
        )
        m.scatter(
            tx,
            ty,
            s=model_track_marker_size,
            facecolor=model_track_marker_face,
            edgecolor=model_track_marker_edge,
            linewidth=0.8,
            zorder=7,
        )
        m.scatter(
            tx[0],
            ty[0],
            s=model_track_marker_size * 1.5,
            facecolor="white",
            edgecolor=model_track_color,
            linewidth=1.5,
            zorder=8,
            label="WRF start",
        )
        m.scatter(
            tx[-1],
            ty[-1],
            s=model_track_marker_size * 1.5,
            facecolor=model_track_marker_face,
            edgecolor="black",
            linewidth=1.0,
            zorder=8,
            label="WRF end",
        )

    if observed_rows:
        observed_edge = observed_track_marker_edge or observed_track_color
        ox, oy = m(obs_lons, obs_lats)
        m.plot(
            ox,
            oy,
            color=observed_track_color,
            linewidth=observed_track_linewidth,
            linestyle=observed_track_linestyle,
            zorder=9,
            label="Observed track",
        )
        m.scatter(
            ox,
            oy,
            s=observed_track_marker_size,
            facecolor=observed_track_marker_face,
            edgecolor=observed_edge,
            linewidth=0.8,
            zorder=10,
        )
        m.scatter(
            ox[0],
            oy[0],
            s=observed_track_marker_size * 1.4,
            facecolor="white",
            edgecolor=observed_track_color,
            linewidth=1.4,
            zorder=11,
            label="Observed start",
        )
        m.scatter(
            ox[-1],
            oy[-1],
            s=observed_track_marker_size * 1.4,
            facecolor=observed_track_marker_face,
            edgecolor="black",
            linewidth=1.0,
            zorder=11,
            label="Observed end",
        )

    ax.set_title(
        f"WRF and observed Typhoon tracks from {len(track_rows)} wrfout_d01 files\n"
        f"Shading: {shading_var}; reference file: {first_file.name}"
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    plt.tight_layout()

    if track_output_file:
        with Path(track_output_file).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "lat", "lon"])
            for row in track_rows:
                writer.writerow([row["time"], f"{row['latitude']:.6f}", f"{row['longitude']:.6f}"])

    if diagnostics_output_file:
        columns = sorted({key for row in track_rows for key in row.keys()})
        with Path(diagnostics_output_file).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(track_rows)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if return_data:
        return fig, track_rows, observed_rows
    return fig

def plot_vertical_cross_section(
    files,
    file_index=0,
    cross_section_axis="lat",
    section_value=19.2,
    lev1=0,
    lev2=30,
    shading_var="W",
    contour_var="QVAPOR",
    vector_horizontal_var="auto",
    vector_vertical_var="W",
    time_index=0,
    vertical_axis="height_km",
    shading_levels=None,
    contour_levels=None,
    shading_level_count=17,
    contour_level_count=13,
    shading_symmetric_about_zero=True,
    contour_symmetric_about_zero=True,
    variable_scale=None,
    variable_units_override=None,
    negative_shade_color=(0.0, 0.15, 0.85),
    positive_shade_color=(0.85, 0.0, 0.0),
    neutral_color=(1.0, 1.0, 1.0),
    negative_contour_color="blue",
    positive_contour_color="red",
    zero_contour_color="black",
    negative_contour_linestyle="dotted",
    positive_contour_linestyle="solid",
    contour_linewidth=1.1,
    contour_label_fontsize=8,
    plot_vectors=True,
    vector_skip_horizontal=6,
    vector_skip_vertical=2,
    vertical_wind_display_multiplier=10.0,
    vector_scale=250,
    vector_width=0.002,
    vector_alpha=0.78,
    vector_key_horizontal_value=20,
    figsize=(11, 6.5),
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    x_tick_count=None,
    y_tick_count=None,
    filename_time_regex=DEFAULT_TIME_REGEX,
    filename_time_format=DEFAULT_TIME_FORMAT,
    save_path=None,
    return_data=False,
):
    """Plot a WRF vertical cross section with shading, contours, and wind vectors."""
    wrf_files = _sort_files(files, filename_time_regex, filename_time_format)
    if not wrf_files:
        raise ValueError("files is empty")
    infile_path = wrf_files[file_index]

    default_scale = {
        "QVAPOR": 1000.0,
        "QCLOUD": 1000.0,
        "QRAIN": 1000.0,
        "QICE": 1000.0,
        "QSNOW": 1000.0,
        "QGRAUP": 1000.0,
    }
    if variable_scale:
        default_scale.update(variable_scale)
    variable_scale = default_scale

    default_units = {
        "QVAPOR": "g kg-1",
        "QCLOUD": "g kg-1",
        "QRAIN": "g kg-1",
        "QICE": "g kg-1",
        "QSNOW": "g kg-1",
        "QGRAUP": "g kg-1",
        "pressure_hpa": "hPa",
        "height_km": "km",
        "wind_speed": "m s-1",
    }
    if variable_units_override:
        default_units.update(variable_units_override)
    variable_units_override = default_units

    with _xr().open_dataset(infile_path) as ds:
        nx, ny, nz = _mass_grid_dimensions(ds)
        lats, lons = _lat_lon_arrays(ds, time_index)
        section_index = _nearest_section_index(cross_section_axis, section_value, lats, lons)

        if cross_section_axis.lower() == "lat":
            actual_value = float(lats[section_index, lats.shape[1] // 2])
            section_lats = lats[section_index, :]
            section_lons = lons[section_index, :]
            horizontal_label = "Distance along longitude section (km)"
            plane_label = f"latitude {actual_value:.3f} deg"
        else:
            actual_value = float(lons[lons.shape[0] // 2, section_index])
            section_lats = lats[:, section_index]
            section_lons = lons[:, section_index]
            horizontal_label = "Distance along latitude section (km)"
            plane_label = f"longitude {actual_value:.3f} deg"

        lev1 = max(0, int(lev1))
        lev2 = min(int(lev2), nz)
        if lev2 <= lev1:
            raise ValueError(f"Invalid layer range lev1:lev2 = {lev1}:{lev2}")
        lev_slice = slice(lev1, lev2)

        shade3d, shade_units = _variable_3d(ds, shading_var, time_index, variable_scale, variable_units_override)
        contour3d, contour_units = _variable_3d(ds, contour_var, time_index, variable_scale, variable_units_override)
        height3d = _height_mass_km(ds, time_index)

        shade_xs = _extract_section(shade3d, cross_section_axis, section_index)[lev_slice, :]
        contour_xs = _extract_section(contour3d, cross_section_axis, section_index)[lev_slice, :]
        height_xs = _extract_section(height3d, cross_section_axis, section_index)[lev_slice, :]

        if vertical_axis == "model_level":
            zcoord = np.repeat(np.arange(lev1, lev2)[:, None], shade_xs.shape[1], axis=1)
            vertical_label = "Model level k"
        else:
            zcoord = height_xs
            vertical_label = "Height (km)"

        xcoord = _cumulative_distance_km(section_lats, section_lons)
        xgrid = np.repeat(xcoord[None, :], shade_xs.shape[0], axis=0)

        if vector_horizontal_var == "auto":
            horizontal_wind_name = "U" if cross_section_axis.lower() == "lat" else "V"
        else:
            horizontal_wind_name = vector_horizontal_var.upper()
        if horizontal_wind_name not in ("U", "V"):
            raise ValueError('vector_horizontal_var must be "auto", "U", or "V"')

        wind_h_3d, _ = _variable_3d(ds, horizontal_wind_name, time_index, variable_scale, variable_units_override)
        wind_w_3d, _ = _variable_3d(ds, vector_vertical_var.upper(), time_index, variable_scale, variable_units_override)
        wind_h_xs = _extract_section(wind_h_3d, cross_section_axis, section_index)[lev_slice, :]
        wind_w_xs = _extract_section(wind_w_3d, cross_section_axis, section_index)[lev_slice, :] * vertical_wind_display_multiplier
        time_label = _wrf_time_label(ds, time_index, fallback=_file_time_label(infile_path, filename_time_regex, filename_time_format))

    if shading_levels is None:
        shading_levels = _make_signed_levels(shade_xs, shading_level_count, shading_symmetric_about_zero)
    else:
        shading_levels = np.asarray(shading_levels, dtype=float)
    if contour_levels is None:
        contour_levels = _make_signed_levels(contour_xs, contour_level_count, contour_symmetric_about_zero)
    else:
        contour_levels = np.asarray(contour_levels, dtype=float)

    shade_cmap, shade_norm = _signed_shading_cmap(
        shading_levels,
        negative_shade_color=negative_shade_color,
        positive_shade_color=positive_shade_color,
        neutral_color=neutral_color,
    )

    fig, ax = plt.subplots(figsize=figsize)
    shaded = ax.contourf(xgrid, zcoord, shade_xs, levels=shading_levels, cmap=shade_cmap, norm=shade_norm, extend="both")
    cbar = fig.colorbar(shaded, ax=ax, pad=0.02)
    cbar.set_label(f"{shading_var} ({shade_units})" if shade_units else shading_var)

    _draw_signed_contours(
        ax,
        xgrid,
        zcoord,
        contour_xs,
        contour_levels,
        negative_color=negative_contour_color,
        positive_color=positive_contour_color,
        zero_color=zero_contour_color,
        negative_linestyle=negative_contour_linestyle,
        positive_linestyle=positive_contour_linestyle,
        linewidth=contour_linewidth,
        label_fontsize=contour_label_fontsize,
    )

    if plot_vectors:
        skip = (
            slice(None, None, max(1, int(vector_skip_vertical))),
            slice(None, None, max(1, int(vector_skip_horizontal))),
        )
        q = ax.quiver(
            xgrid[skip],
            zcoord[skip],
            wind_h_xs[skip],
            wind_w_xs[skip],
            scale=vector_scale,
            width=vector_width,
            alpha=vector_alpha,
            color="black",
        )
        ax.quiverkey(
            q,
            0.86,
            1.04,
            vector_key_horizontal_value,
            f"{vector_key_horizontal_value:g} m/s horizontal; W x {vertical_wind_display_multiplier:g}",
            labelpos="E",
            coordinates="axes",
        )

    contour_legend = [
        Line2D([0], [0], color=negative_contour_color, linestyle=negative_contour_linestyle, lw=contour_linewidth, label=f"negative {contour_var}"),
        Line2D([0], [0], color=positive_contour_color, linestyle=positive_contour_linestyle, lw=contour_linewidth, label=f"positive {contour_var}"),
    ]
    ax.legend(handles=contour_legend, loc="upper right")
    ax.set_title(f"WRF vertical cross section at {plane_label}\n{infile_path.name}, time={time_label}")
    _set_axis_labels_and_ticks(
        ax,
        xlabel=horizontal_label,
        ylabel=vertical_label,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        x_tick_count=x_tick_count,
        y_tick_count=y_tick_count,
    )
    ax.grid(True, linewidth=0.25, alpha=0.35)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if return_data:
        data = {
            "xgrid": xgrid,
            "zcoord": zcoord,
            "shade_xs": shade_xs,
            "contour_xs": contour_xs,
            "wind_h_xs": wind_h_xs,
            "wind_w_xs": wind_w_xs,
            "shading_levels": shading_levels,
            "contour_levels": contour_levels,
            "section_index": section_index,
            "plane_label": plane_label,
            "file": str(infile_path),
        }
        return fig, data
    return fig


__all__ = [
    "parse_observed_atcf_track",
    "plot_animation_d03",
    "plot_track",
    "plot_vertical_cross_section",
]
