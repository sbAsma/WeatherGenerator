# (C) Copyright 2025 WeatherGenerator contributors.
# Licensed under Apache 2.0.

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from weathergen.common.io import CAMSForecastReader
from weathergen.evaluate.io.wegen_reader import WeatherGenZarrReader
from weathergen.evaluate.plotting.plotter import Plotter

_logger = logging.getLogger(__name__)

# Number of parallel workers for data loading and rendering
_N_WORKERS = 8

# ---------------------------------------------------------------------------
# PPM conversion helpers
# ---------------------------------------------------------------------------

# Molecular weight of dry air (g/mol)
_M_AIR: float = 28.97

# Species molecular weights (g/mol), keyed by channel-name prefix.
# Profile channels follow the pattern ``<species>_<level>`` (e.g. ``co_500``).
_M_SPECIES: dict[str, float] = {
    "co": 28.01,
    "no2": 46.01,
    "no": 30.01,
    "so2": 64.07,
    "o3": 48.00,
    "go3": 48.00,
}


def _channel_ppm_factor(channel: str) -> float | None:
    """Return the kg kg\u207b\u00b9 \u2192 ppmv conversion factor for *channel*, or ``None``.

    Profile channels (e.g. ``co_500``) are converted using
    ``ppmv = value * (M_air / M_species) * 1e6``.
    Total-column (``tc_*``) and surface-particulate (``pm*``) channels are
    excluded because ppmv is not a meaningful unit for them.
    """
    ch = channel.lower()
    if ch.startswith("tc_") or ch.startswith("pm"):
        return None
    # match longest prefix first (no2 before no)
    for species in sorted(_M_SPECIES, key=len, reverse=True):
        if ch.startswith(species + "_") or ch == species:
            return (_M_AIR / _M_SPECIES[species]) * 1e6
    return None


def _compute_rmse(pred: xr.DataArray, target: xr.DataArray, dim: str = "ipoint") -> xr.DataArray:
    """Compute root-mean-square error along *dim*."""
    return np.sqrt(((pred - target) ** 2).mean(dim=dim))


# Maximum allowed gap (in hours) between the requested CAMS init time and
# the nearest available time coordinate.  If the gap exceeds this threshold
# a warning is logged during verification and the data is flagged.
_MAX_TIME_GAP_HOURS: int = 6


def _verify_cams_time_selection(
    da: xr.DataArray,
    init_time: "np.datetime64",
    channel: str,
    forecast_hour: int,
) -> None:
    """Warn when the CAMS time coordinate doesn't closely match *init_time*.

    Exact time selection should be used for CAMS slicing; if this helper is
    called, it reports the selected-vs-requested gap so large offsets are
    immediately visible.
    """
    if "time" not in da.coords:
        return
    selected_time = np.datetime64(da["time"].values, "ns")
    requested_time = np.datetime64(init_time, "ns")
    gap = abs(int((selected_time - requested_time) / np.timedelta64(1, "h")))
    if gap > _MAX_TIME_GAP_HOURS:
        _logger.warning(
            "CAMS time mismatch for channel %s at forecast hour %dh: "
            "requested init_time=%s but nearest available is %s (gap=%dh). "
            "This may indicate the CAMS dataset does not cover the evaluation period.",
            channel, forecast_hour,
            _format_datetime64(requested_time),
            _format_datetime64(selected_time),
            gap,
        )
    else:
        _logger.debug(
            "CAMS time OK for %s fstep %dh: selected %s (gap=%dh)",
            channel, forecast_hour,
            _format_datetime64(selected_time), gap,
        )


def _verify_cams_data_consistency(
    cams_cache: dict[tuple[str, int], np.ndarray],
    channels: list[str],
    forecast_hours: list[int],
    hr_info: dict[int, tuple],
) -> None:
    """Run post-load sanity checks on the pre-loaded CAMS data.

    Checks performed:
    1. **Per-(channel, hour) statistics** – mean, std, min, max are logged so
       anomalous orders of magnitude are visible.
    2. **Cross-step variance** – if the CAMS values for a channel barely
       change across forecast steps, the data may have been selected
       incorrectly (e.g. the same time slice reused for every step).
    3. **NaN fraction** – a high NaN ratio indicates interpolation issues.
    4. **Order-of-magnitude vs WG target** – large discrepancies hint at
       unit mismatches or wrong variable selection.
    """
    _logger.info("Running CAMS data consistency checks …")

    for ch in channels:
        means_across_steps: list[float] = []
        for hr in forecast_hours:
            key = (ch, hr)
            if key not in cams_cache:
                _logger.warning("  Missing CAMS data for %s at %dh", ch, hr)
                continue
            vals = cams_cache[key]
            nan_frac = float(np.isnan(vals).mean())
            vmean = float(np.nanmean(vals))
            vstd = float(np.nanstd(vals))
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            means_across_steps.append(vmean)

            _logger.info(
                "  CAMS %-12s fstep %3dh: mean=%.4e  std=%.4e  "
                "min=%.4e  max=%.4e  NaN%%=%.1f",
                ch, hr, vmean, vstd, vmin, vmax, nan_frac * 100,
            )
            if nan_frac > 0.1:
                _logger.warning(
                    "  HIGH NaN fraction (%.1f%%) for %s at %dh – "
                    "interpolation may be unreliable.",
                    nan_frac * 100, ch, hr,
                )

            # compare with WG target order of magnitude
            if hr in hr_info:
                _raw, wg_target, _wg_pred, _init = hr_info[hr]
                tar = np.asarray(wg_target.sel(channel=ch).values).ravel()
                tar_mean = float(np.nanmean(tar))
                if tar_mean != 0 and vmean != 0:
                    ratio = abs(vmean / tar_mean)
                    if ratio > 100 or ratio < 0.01:
                        _logger.warning(
                            "  ORDER-OF-MAGNITUDE mismatch for %s at %dh: "
                            "CAMS mean=%.4e vs WG target mean=%.4e (ratio=%.2g). "
                            "Check units/variable selection.",
                            ch, hr, vmean, tar_mean, ratio,
                        )

        # cross-step variance check
        if len(means_across_steps) > 1:
            step_std = float(np.std(means_across_steps))
            step_mean = float(np.mean(np.abs(means_across_steps)))
            if step_mean > 0 and step_std / step_mean < 0.001:
                _logger.warning(
                    "  SUSPICIOUS: CAMS %s mean is nearly constant across %d "
                    "forecast steps (std/|mean|=%.2e). The same time slice may "
                    "have been selected for every step.",
                    ch, len(means_across_steps), step_std / step_mean,
                )

    _logger.info("CAMS data consistency checks complete.")


def _preload_and_interpolate_cams(
    cams_reader: "CAMSForecastReader",
    channels: list[str],
    forecast_hours: list[int],
    hr_info: dict[int, tuple],
    n_workers: int = _N_WORKERS,
) -> dict[tuple[str, int], np.ndarray]:
    """Pre-load CAMS data into memory and batch-interpolate all channels per hour.

    This avoids repeated dask/zarr reads and creates a single
    ``RegularGridInterpolator`` per forecast hour (instead of one per
    channel × hour), giving a large speed-up.

    Returns
    -------
    dict[(channel, hour), np.ndarray]
        Interpolated CAMS values (1-D, at WG scatter points) for every
        (channel, hour) pair.
    """
    _logger.info(
        "Pre-loading CAMS data: %d channels × %d forecast hours …",
        len(channels), len(forecast_hours),
    )

    # ---- 1. identify the base xarray variables we need ------------------
    needed_vars: set[str] = set()
    for ch in channels:
        if ch in cams_reader.ds.data_vars:
            needed_vars.add(ch)
        else:
            var = ch.split("_")[0]
            if var in cams_reader.ds.data_vars:
                needed_vars.add(var)

    # ---- 2. load the subset eagerly (dask → numpy) ----------------------
    # Subset along step, level, and time *before* loading to avoid OOM
    # when the full CAMS dataset is too large for available memory.
    subset = cams_reader.ds[list(needed_vars)]

    if "step" in subset.dims:
        needed_steps = [pd.Timedelta(hours=int(hr)) for hr in forecast_hours]
        subset = subset.sel(step=needed_steps)

    needed_levels = sorted({
        int(ch.split("_")[-1])
        for ch in channels
        if "_" in ch and ch.split("_")[-1].isdigit()
    })
    if "isobaricInhPa" in subset.dims and needed_levels:
        subset = subset.sel(isobaricInhPa=needed_levels)

    init_times = sorted({
        hr_info[hr][3]
        for hr in forecast_hours
        if hr in hr_info and hr_info[hr][3] is not None
    })
    if "time" in subset.dims and init_times:
        requested_times = np.array(init_times, dtype="datetime64[ns]")
        subset = subset.sel(time=requested_times)

    subset = subset.load()
    _logger.info("CAMS dataset loaded into memory (%d variables).", len(needed_vars))

    # ---- 3. per-hour batch interpolation --------------------------------
    cams_lat = cams_reader.lat
    cams_lon = cams_reader.lon

    def _process_hour(hr: int) -> dict[tuple[str, int], np.ndarray]:
        if hr not in hr_info:
            return {}
        _raw, wg_target, _wg_pred, init_time = hr_info[hr]

        # target coords from WG (identical for every channel in the stream)
        sample_tar = wg_target.sel(channel=channels[0])
        target_lat = sample_tar["lat"].values
        target_lon = sample_tar["lon"].values
        target_points = np.column_stack([target_lat, target_lon])

        # extract sorted 2-D arrays for every channel
        raw_arrays: list[np.ndarray] = []
        for ch in channels:
            if ch in subset.data_vars:
                da = subset[ch]
            else:
                ch_parts = ch.split("_")
                var, level = ch_parts[0], ch_parts[1]
                da = subset[var].sel(isobaricInhPa=level)

            if "step" in da.dims:
                da = da.sel(step=pd.Timedelta(hours=int(hr)))
            if "time" in da.dims:
                if init_time is not None:
                    da = da.sel(time=init_time)
                elif da.sizes.get("time", 0) > 1:
                    _logger.warning(
                        "No init_time for %s fstep %dh – falling back to "
                        "first time entry; results may be incorrect.",
                        ch, hr,
                    )
                    da = da.isel(time=0)

            data = cams_reader._sort_data(da.values)
            while data.ndim > 2:
                data = data[0]
            raw_arrays.append(data)

        # batch interpolation: stack channels as extra dimension
        stacked = np.stack(raw_arrays, axis=-1)  # (lat, lon, n_channels)
        interp = RegularGridInterpolator(
            (cams_lat, cams_lon),
            stacked,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        result = interp(target_points)  # (n_points, n_channels)

        return {(ch, hr): result[:, ci] for ci, ch in enumerate(channels)}

    # run hours in parallel – scipy interpolation releases the GIL
    hour_cache: dict[tuple[str, int], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, len(forecast_hours))) as pool:
        futures = {pool.submit(_process_hour, hr): hr for hr in forecast_hours}
        for fut in as_completed(futures):
            hr_done = futures[fut]
            hour_cache.update(fut.result())
            _logger.info("  Interpolated CAMS data for forecast hour %dh", hr_done)

    _logger.info(
        "CAMS pre-load complete: %d (channel, hour) pairs ready.",
        len(hour_cache),
    )

    # Run data consistency checks on the interpolated cache
    _verify_cams_data_consistency(hour_cache, channels, forecast_hours, hr_info)

    return hour_cache


def _compute_cams_native_rmse(
    cams_reader: "CAMSForecastReader",
    analysis_path: "Path",
    channels: list[str],
    forecast_hours: list[int],
    hr_info: dict[int, tuple],
) -> dict[tuple[str, int], float]:
    """Compute CAMS forecast-vs-analysis RMSE on the **native CAMS grid**.

    This matches the standalone scorecard script: for each (channel, hour)
    the CAMS forecast (at init_time, step=hour) is compared against the
    CAMS analysis (at valid_time = init_time + hour) on the full CAMS grid.
    If the forecast and analysis grids differ the forecast is regridded to
    the analysis grid before computing RMSE.

    Returns
    -------
    dict[(channel, hour), float]
        Scalar RMSE values per (channel, hour) pair.
    """
    _logger.info(
        "Computing CAMS forecast-vs-analysis RMSE on native grid: "
        "%d channels × %d forecast hours …",
        len(channels), len(forecast_hours),
    )

    # ---- open analysis zarr ---------------------------------------------
    ds_a_surface = xr.open_zarr(analysis_path, group="surface", chunks="auto")
    ds_a_profiles = xr.open_zarr(analysis_path, group="profiles", chunks="auto")
    ds_analysis = xr.merge([ds_a_surface, ds_a_profiles])

    # ---- pre-select forecast subset for efficiency ----------------------
    ds_forecast = cams_reader.ds

    rmse_out: dict[tuple[str, int], float] = {}

    for hr in forecast_hours:
        if hr not in hr_info:
            continue
        _, _, _, init_time = hr_info[hr]
        if init_time is None:
            continue
        valid_time = np.datetime64(init_time, "ns") + np.timedelta64(int(hr), "h")

        # Select exact time slices
        ds_f = ds_forecast
        ds_a = ds_analysis
        if "time" in ds_f.dims:
            ds_f = ds_f.sel(time=init_time)
        if "time" in ds_a.dims:
            ds_a = ds_a.sel(time=valid_time)

        for ch in channels:
            # ---- resolve variable + level --------------------------------
            if ch in ds_f.data_vars and ch in ds_a.data_vars:
                forecast_da = ds_f[ch]
                analysis_da = ds_a[ch]
            else:
                parts = ch.split("_")
                var = parts[0]
                level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if var not in ds_f.data_vars or var not in ds_a.data_vars:
                    _logger.warning("Variable %s not in both datasets – skipping %s", var, ch)
                    continue
                forecast_da = ds_f[var]
                analysis_da = ds_a[var]
                if level is not None and "isobaricInhPa" in forecast_da.dims:
                    forecast_da = forecast_da.sel(isobaricInhPa=level)
                    analysis_da = analysis_da.sel(isobaricInhPa=level)

            # ---- select forecast step ------------------------------------
            if "step" in forecast_da.dims:
                forecast_da = forecast_da.sel(step=pd.Timedelta(hours=int(hr)))

            # ---- squeeze leftover scalar dims ----------------------------
            if "time" in forecast_da.dims:
                forecast_da = forecast_da.squeeze("time")
            if "time" in analysis_da.dims:
                analysis_da = analysis_da.squeeze("time")

            # ---- normalise longitude to [-180, 180] ----------------------
            for da_ref in (forecast_da, analysis_da):
                if da_ref.longitude.max() > 180:
                    da_ref = da_ref.assign_coords(
                        longitude=(((da_ref.longitude + 180) % 360) - 180)
                    ).sortby("longitude")

            forecast_vals = forecast_da.values
            analysis_vals = analysis_da.values

            # ---- collapse extra leading dims -----------------------------
            while forecast_vals.ndim > 2:
                forecast_vals = forecast_vals[0]
            while analysis_vals.ndim > 2:
                analysis_vals = analysis_vals[0]

            # ---- regrid forecast → analysis if shapes differ -------------
            if forecast_vals.shape != analysis_vals.shape:
                f_lat = forecast_da.latitude.values
                f_lon = forecast_da.longitude.values
                a_lat = analysis_da.latitude.values
                a_lon = analysis_da.longitude.values

                interpolator = RegularGridInterpolator(
                    (f_lat, f_lon),
                    forecast_vals,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                a_lon_grid, a_lat_grid = np.meshgrid(a_lon, a_lat)
                pts = np.column_stack([a_lat_grid.ravel(), a_lon_grid.ravel()])
                forecast_vals = interpolator(pts).reshape(len(a_lat), len(a_lon))

            rmse_val = float(np.sqrt(((forecast_vals - analysis_vals) ** 2).mean()))
            rmse_out[(ch, hr)] = rmse_val

    _logger.info(
        "CAMS native-grid RMSE complete: %d (channel, hour) pairs computed.",
        len(rmse_out),
    )
    return rmse_out


def _scorecard_channel_order(channel: str) -> tuple[int, str, float | str]:
    """Return a stable sort key for scorecard heatmap channels.

    Pressure-level channels such as ``co_50`` and ``co_1000`` are ordered by
    ascending level so they render top-to-bottom as 50, ..., 1000. Total-column
    channels such as ``tc_co`` are placed after the pressure levels.
    """
    ch = channel.lower()

    if ch.startswith("tc_"):
        return (2, ch[3:], float("inf"))

    species, sep, suffix = ch.rpartition("_")
    if sep and suffix.isdigit():
        return (0, species, float(suffix))

    return (1, ch, ch)



def _cams_steps_as_hours(cams_reader: CAMSForecastReader) -> set[int]:
    """Return CAMS forecast steps converted to integer hours.

    CAMS stores ``step`` as ``numpy.timedelta64``; converting via
    ``pd.Timedelta`` gives us reliable hour values.
    """
    import pandas as pd

    hours: set[int] = set()
    for s in cams_reader.forecast_steps:
        hours.add(int(pd.Timedelta(s).total_seconds() // 3600))
    return hours


def _align_forecast_steps(
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    requested: list[int],
) -> tuple[list[int], dict[int, int]]:
    """Return intersection of forecast steps converted to hours.

    WG reader steps are integer *indices* (0, 1, 2, …).  Each index
    corresponds to ``index * step_hrs`` hours (``step_hrs`` comes from the
    WG inference config, e.g. 6 for CAMS).  CAMS forecast steps are stored
    as ``numpy.timedelta64`` and are converted to integer hours here.

    Parameters
    ----------
    requested: list[int]
        Forecast steps expressed in **hours** (or the word ``"all"``,
        handled by the caller before reaching this function).

    Returns
    -------
    common_hours : list[int]
        Sorted list of hours present in both WG and CAMS datasets.
    wg_map : dict[int, int]
        Mapping from hour -> original WG step index so that subsequent
        reader calls can use the raw value.
    """
    step_hrs = wg_reader.step_hrs  # hours per WG index
    wg_map: dict[int, int] = {}
    for idx in wg_reader.get_forecast_steps():
        idx = int(idx)
        hr = idx * step_hrs
        wg_map[hr] = idx

    available_wg = set(wg_map.keys())
    available_cams = _cams_steps_as_hours(cams_reader)

    _logger.info("WG available hours: %s", sorted(available_wg))
    _logger.info("CAMS available hours: %s", sorted(available_cams))

    common = sorted(available_wg & available_cams & set(requested))
    missing_wg = set(requested) - available_wg
    missing_cams = set(requested) - available_cams
    if missing_wg:
        _logger.warning(
            "Forecast steps (hours) %s not found in WG output; they will be skipped",
            sorted(missing_wg),
        )
    if missing_cams:
        _logger.warning(
            "Forecast steps (hours) %s not available in CAMS data; they will be skipped",
            sorted(missing_cams),
        )
    return common, wg_map


def _format_datetime64(value: np.datetime64) -> str:
    """Return a stable string representation for datetime64 values."""
    return pd.Timestamp(value).isoformat()


def _extract_unique_valid_times(data: xr.DataArray, label: str) -> np.ndarray:
    """Return sorted non-NaT valid_time values for a WG field."""
    if "valid_time" not in data.coords:
        raise ValueError(
            f"{label} is missing the valid_time coordinate required for CAMS comparison."
        )

    valid_time = np.asarray(data["valid_time"].values).reshape(-1)
    if valid_time.size == 0:
        raise ValueError(f"{label} has no valid_time values to verify.")

    valid_time = valid_time.astype("datetime64[ns]", copy=False)
    valid_time = valid_time[~np.isnat(valid_time)]
    if valid_time.size == 0:
        raise ValueError(f"{label} only contains NaT valid_time values.")

    return np.unique(valid_time)


def _cams_expected_valid_time(
    cams_reader: CAMSForecastReader,
    forecast_hour: int,
    init_time: "np.datetime64 | None" = None,
) -> np.datetime64:
    """Return the CAMS valid time for *forecast_hour*.

    When *init_time* is supplied the valid time is simply
    ``init_time + forecast_hour``.  Otherwise the first forecast
    initialisation in the CAMS dataset is used (legacy behaviour).
    """
    if init_time is not None:
        return np.datetime64(init_time, "ns") + np.timedelta64(int(forecast_hour), "h")

    init_times = getattr(cams_reader, "_wg_cached_init_times", None)
    if init_times is None:
        init_times = np.asarray(cams_reader.time).reshape(-1)
        init_times = init_times.astype("datetime64[ns]", copy=False)
        init_times = np.unique(init_times[~np.isnat(init_times)])
        setattr(cams_reader, "_wg_cached_init_times", init_times)

    if init_times.size == 0:
        raise ValueError(
            f"CAMS dataset {cams_reader.cams_forecast_path} has no valid forecast initial time."
        )

    if init_times.size > 1 and not getattr(cams_reader, "_wg_warned_multi_time", False):
        _logger.warning(
            "CAMS dataset %s contains %d forecast initial times; comparison uses the first one (%s).",
            cams_reader.cams_forecast_path,
            init_times.size,
            _format_datetime64(init_times[0]),
        )
        setattr(cams_reader, "_wg_warned_multi_time", True)

    return init_times[0] + np.timedelta64(int(forecast_hour), "h")


def _verify_timestep_alignment(
    wg_target: xr.DataArray,
    wg_pred: xr.DataArray,
    cams_reader: CAMSForecastReader,
    forecast_hour: int,
) -> np.datetime64:
    """Ensure WG target/prediction and CAMS timestamps refer to the same valid time."""
    target_times = _extract_unique_valid_times(
        wg_target, f"WG target data for forecast hour {forecast_hour}h"
    )
    pred_times = _extract_unique_valid_times(
        wg_pred, f"WG prediction data for forecast hour {forecast_hour}h"
    )

    if target_times.size != 1:
        formatted = [_format_datetime64(ts) for ts in target_times]
        raise ValueError(
            f"WG target data for forecast hour {forecast_hour}h spans multiple valid_time "
            f"values: {formatted}. CAMS comparison expects a single timestamp per forecast hour."
        )

    if pred_times.size != 1:
        formatted = [_format_datetime64(ts) for ts in pred_times]
        raise ValueError(
            f"WG prediction data for forecast hour {forecast_hour}h spans multiple valid_time "
            f"values: {formatted}. CAMS comparison expects a single timestamp per forecast hour."
        )

    wg_valid_time = target_times[0]
    if pred_times[0] != wg_valid_time:
        raise ValueError(
            "WG target/prediction timestep mismatch for forecast hour "
            f"{forecast_hour}h: target={_format_datetime64(wg_valid_time)}, "
            f"prediction={_format_datetime64(pred_times[0])}."
        )

    # Derive the expected CAMS init time from the WG valid time so that
    # multi-init-time CAMS datasets are handled correctly.
    init_time = wg_valid_time - np.timedelta64(int(forecast_hour), "h")
    cams_valid_time = _cams_expected_valid_time(cams_reader, forecast_hour, init_time=init_time)
    if wg_valid_time != cams_valid_time:
        raise ValueError(
            "WG and CAMS timesteps are misaligned for forecast hour "
            f"{forecast_hour}h: WG valid_time={_format_datetime64(wg_valid_time)}, "
            f"CAMS expected valid_time={_format_datetime64(cams_valid_time)}."
        )

    return wg_valid_time


def _prepare_bias_da(pred: xr.DataArray, tar: xr.DataArray, cams_vals: np.ndarray):
    # return wg_bias_da, cams_bias_da computed from inputs
    lat = tar['lat'].values
    lon = tar['lon'].values
    wg_bias = pred - tar
    cams_vals = np.asarray(cams_vals).ravel()
    wg_bias_flat = np.asarray(wg_bias).ravel()
    tar_flat = np.asarray(tar.values).ravel()
    lat_flat = np.asarray(lat).ravel()
    lon_flat = np.asarray(lon).ravel()
    _logger.info(f"[DEBUG] lat length: {lat_flat.size}, lon length: {lon_flat.size}")
    _logger.info(f"[DEBUG] wg_bias_flat shape: {wg_bias_flat.shape}, wg_bias_flat size: {wg_bias_flat.size}")
    _logger.info(f"[DEBUG] tar_flat shape: {tar_flat.shape}")
    _logger.info(f"[DEBUG] cams_vals_flat shape: {cams_vals.shape}")
    if wg_bias_flat.size == lat_flat.size * lon_flat.size:
        wg_bias_grid = wg_bias_flat.reshape(len(lat_flat), len(lon_flat))
        cams_bias = (cams_vals - tar_flat).reshape(len(lat_flat), len(lon_flat))
        wg_bias_da = xr.DataArray(wg_bias_grid, coords={'lat': lat_flat, 'lon': lon_flat}, dims=['lat','lon'])
        cams_bias_da = xr.DataArray(cams_bias, coords={'lat': lat_flat, 'lon': lon_flat}, dims=['lat','lon'])
    else:
        if not (wg_bias_flat.size == lat_flat.size == lon_flat.size):
            _logger.error(f"wg_bias shape {wg_bias_flat.shape} does not match lat/lon shapes {lat_flat.shape}/{lon_flat.shape}.")
            raise ValueError(f"wg_bias shape {wg_bias_flat.shape} does not match lat/lon shapes {lat_flat.shape}/{lon_flat.shape}.")
        ipoints = np.arange(wg_bias_flat.size)
        wg_bias_da = xr.DataArray(wg_bias_flat, coords={'ipoint': ipoints, 'lat':('ipoint', lat_flat), 'lon':('ipoint', lon_flat)}, dims=['ipoint'])
        cams_bias_da = xr.DataArray(cams_vals - tar_flat, coords={'ipoint': ipoints, 'lat':('ipoint', lat_flat), 'lon':('ipoint', lon_flat)}, dims=['ipoint'])
    return wg_bias_da, cams_bias_da


def _render_bias_frame(
    ch: str,
    hr: int,
    wg_bias_vals: np.ndarray,
    cams_bias_vals: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    maxabs: float,
    convert_to_ppm: bool,
    out_path: "Path",
) -> "Path":
    """Render a single bias comparison frame (thread-safe, no pyplot)."""
    import cartopy.crs as ccrs
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 8), dpi=300)
    FigureCanvasAgg(fig)

    ax_cams = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax_wg = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    sc = None
    for ax, vals, title in [
        (ax_cams, cams_bias_vals, "CAMS Forecast \u2013 Analysis"),
        (ax_wg, wg_bias_vals, "WG Prediction \u2013 Target"),
    ]:
        ax.coastlines()
        sc = ax.scatter(
            lon, lat, c=vals,
            cmap="coolwarm", vmin=-maxabs, vmax=maxabs,
            transform=ccrs.PlateCarree(), s=1, linewidths=0.0,
        )
        ax.set_global()
        ax.set_title(f"{title} \u2013 {ch} (fstep {hr})")

    bias_unit = "ppm" if (convert_to_ppm and _channel_ppm_factor(ch) is not None) else "kg kg\u207b\u00b9"
    cbar = fig.colorbar(sc, ax=[ax_cams, ax_wg], orientation="horizontal", pad=0.02, fraction=0.04)
    cbar.set_label(f"Bias ({bias_unit})")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
    return out_path


def _render_value_frame(
    ch: str,
    hr: int,
    lat: np.ndarray,
    lon: np.ndarray,
    cams_vals: np.ndarray,
    pred_vals: np.ndarray,
    vmin: float,
    vmax: float,
    value_unit: str,
    out_path: "Path",
) -> "Path":
    """Render a single value comparison frame (thread-safe, no pyplot)."""
    import cartopy.crs as ccrs
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 8), dpi=300)
    FigureCanvasAgg(fig)

    ax_cams = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax_wg = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    last_sc = None
    for ax, title, vals in [
        (ax_cams, "CAMS Forecast", cams_vals),
        (ax_wg, "WG Prediction", pred_vals),
    ]:
        ax.coastlines()
        last_sc = ax.scatter(
            lon, lat, c=vals,
            cmap="viridis", vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), s=1, linewidths=0.0,
        )
        ax.set_global()
        ax.set_title(f"{title} \u2013 {ch} (fstep {hr}h)")

    cbar = fig.colorbar(last_sc, ax=[ax_cams, ax_wg], orientation="horizontal", pad=0.02, fraction=0.04)
    cbar.set_label(f"{ch} ({value_unit})")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
    return out_path


def _plot_bias_maps(
    plotter: Plotter,
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    stream: str,
    channels: list[str],
    forecast_hours: list[int],
    wg_map: dict[int, int],
    run_id: str,
    convert_to_ppm: bool = False,
    color_max_ppb: float | None = None,
    wg_data=None,
    cams_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> None:
    """Produce and save global bias scatter maps for each forecast step/channel.

    The colour scale is held **constant** across all forecast steps for a
    given channel so that animations are meaningful.  An optional
    ``color_max_ppb`` cap (in ppb) limits the symmetric range.

    Maps are written under ``plotter.out_plot_basedir/<stream>/maps/{wg_bias,cams_bias}``.
    """
    n_hours = len(forecast_hours)
    n_channels = len(channels)
    total_frames = n_hours * n_channels
    _logger.info(
        "Bias maps: %d channels x %d forecast hours = %d frames",
        n_channels, n_hours, total_frames,
    )

    # make sure output directories exist before looping
    for tag in ("wg_bias", "cams_bias"):
        outdir = plotter.get_map_output_dir(tag)
        if not outdir.exists():
            _logger.info(f"Creating directory {outdir}")
            outdir.mkdir(parents=True, exist_ok=True)

    combo_dir = plotter.out_plot_basedir / stream / "maps" / "bias_compare"
    combo_dir.mkdir(parents=True, exist_ok=True)

    # ---- first pass: compute per-channel global bias range ---------------
    _logger.info("Bias maps [1/2]: computing per-channel global bias range …")
    channel_maxabs: dict[str, float] = {ch: 0.0 for ch in channels}
    # cache converted bias DataArrays to avoid reloading in the second pass
    bias_cache: dict[tuple[str, int], tuple[xr.DataArray, xr.DataArray]] = {}

    for hi, hr in enumerate(forecast_hours, 1):
        _logger.info("  Computing biases for forecast hour %dh (%d/%d) …", hr, hi, n_hours)
        raw = wg_map[hr]

        # Use pre-loaded WG data when available; otherwise fall back to loading.
        if wg_data is not None:
            if raw not in wg_data.target:
                continue
            wg_target = wg_data.target[raw]
            wg_pred = wg_data.prediction[raw]
        else:
            _wg = wg_reader.get_data(stream=stream, fsteps=[raw], channels=channels)
            if raw not in _wg.target:
                continue
            wg_target = _wg.target[raw]
            wg_pred = _wg.prediction[raw]

        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)
            lat = tar["lat"].values
            lon = tar["lon"].values

            # Use cached CAMS data when available.
            if cams_cache is not None and (ch, hr) in cams_cache:
                cams_vals = cams_cache[(ch, hr)]
            else:
                cams_vals = cams_reader.get_data(ch, step=hr, target_lat=lat, target_lon=lon)
                cams_vals = np.asarray(cams_vals).ravel()
            wg_bias_da, cams_bias_da = _prepare_bias_da(pred, tar, cams_vals)

            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                wg_bias_da = wg_bias_da * ppm_factor
                cams_bias_da = cams_bias_da * ppm_factor

            bias_cache[(ch, hr)] = (wg_bias_da, cams_bias_da)

            all_vals = np.concatenate([
                np.asarray(wg_bias_da).ravel(),
                np.asarray(cams_bias_da).ravel(),
            ])
            if all_vals.size > 0:
                maxabs = float(np.nanmax(np.abs(all_vals)))
                channel_maxabs[ch] = max(channel_maxabs[ch], maxabs)

    # apply colour cap from color_max_ppb (value is in ppb; display may be ppm)
    for ch in channels:
        if color_max_ppb is not None:
            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                cap = color_max_ppb / 1000.0  # ppb -> ppm
            else:
                # convert ppb back to kg/kg for this species
                inv = _channel_ppm_factor(ch)
                cap = (color_max_ppb / 1000.0 / inv) if inv else color_max_ppb
            if channel_maxabs[ch] > cap:
                _logger.warning(
                    f"Bias channel {ch}: max |bias| {channel_maxabs[ch]:.4g} "
                    f"exceeds color cap ({color_max_ppb} ppb = {cap:.4g}); clipping."
                )
            channel_maxabs[ch] = min(channel_maxabs[ch], cap)
        _logger.info(
            f"Bias colour range for {ch}: +/- {channel_maxabs[ch]:.4g}"
        )

    # ---- second pass: plot with fixed colour range (parallel) ------------
    _logger.info("Bias maps [2/2]: rendering %d frames using %d threads \u2026", total_frames, _N_WORKERS)
    render_tasks = []
    for hr in forecast_hours:
        for ch in channels:
            key = (ch, hr)
            if key not in bias_cache:
                continue
            wg_bias_da, cams_bias_da = bias_cache[key]
            render_tasks.append((
                ch, hr,
                np.asarray(wg_bias_da).ravel(),
                np.asarray(cams_bias_da).ravel(),
                np.asarray(wg_bias_da["lat"]).ravel(),
                np.asarray(wg_bias_da["lon"]).ravel(),
                channel_maxabs[ch],
                convert_to_ppm,
                combo_dir / f"bias_{ch}_fstep_{hr:03d}.png",
            ))

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        futures = {
            pool.submit(_render_bias_frame, *args): args[0:2]
            for args in render_tasks
        }
        for fi, future in enumerate(as_completed(futures), 1):
            ch_done, hr_done = futures[future]
            fname = future.result()
            _logger.info(
                "  Saved bias frame %d/%d: %s", fi, len(render_tasks), Path(fname).name,
            )


def _plot_value_maps(
    plotter: Plotter,
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    stream: str,
    channels: list[str],
    forecast_hours: list[int],
    wg_map: dict[int, int],
    run_id: str,
    convert_to_ppm: bool = False,
    color_max_ppb: float | None = None,
    wg_data=None,
    cams_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> "Path":
    """Produce and save side-by-side global maps of CAMS forecast vs WG prediction.

    Each frame is a Robinson-projection scatter plot with two panels:
    *CAMS Forecast* (left) and *WG Prediction* (right), sharing a **constant**
    colour scale across all forecast steps for a given channel.  The range
    is derived from the global min/max across all steps and can optionally
    be capped via ``color_max_ppb`` (in ppb).

    Frames are written under
    ``plotter.out_plot_basedir/<stream>/maps/value_compare/``
    with the naming convention ``values_{ch}_fstep_{hr:03d}.png``.

    Returns
    -------
    Path
        Output directory where the frames were saved.
    """
    n_hours = len(forecast_hours)
    n_channels = len(channels)
    total_frames = n_hours * n_channels
    _logger.info(
        "Value maps: %d channels x %d forecast hours = %d frames",
        n_channels, n_hours, total_frames,
    )

    value_dir = plotter.out_plot_basedir / stream / "maps" / "value_compare"
    value_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Saving value comparison maps to {value_dir}")

    # ---- first pass: collect data and compute per-channel global ranges --
    _logger.info("Value maps [1/2]: collecting data and computing colour ranges …")
    # keyed by (channel, hour) -> (lat, lon, cams_vals, pred_vals)
    frame_data: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    channel_ranges: dict[str, tuple[float, float]] = {}  # ch -> (vmin, vmax)

    for hi, hr in enumerate(forecast_hours, 1):
        _logger.info("  Collecting value data for forecast hour %dh (%d/%d) …", hr, hi, n_hours)
        raw = wg_map[hr]

        # Use pre-loaded WG data when available; otherwise fall back to loading.
        if wg_data is not None:
            if raw not in wg_data.target:
                _logger.warning(
                    f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping."
                )
                continue
            wg_target = wg_data.target[raw]
            wg_pred = wg_data.prediction[raw]
        else:
            _wg = wg_reader.get_data(stream=stream, fsteps=[raw], channels=channels)
            if raw not in _wg.target:
                _logger.warning(
                    f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping."
                )
                continue
            wg_target = _wg.target[raw]
            wg_pred = _wg.prediction[raw]

        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)
            lat = tar["lat"].values
            lon = tar["lon"].values

            # Use cached CAMS data when available.
            if cams_cache is not None and (ch, hr) in cams_cache:
                cams_vals = cams_cache[(ch, hr)]
            else:
                cams_vals = cams_reader.get_data(
                    ch, step=hr, target_lat=lat, target_lon=lon
                )
                cams_vals = np.asarray(cams_vals).ravel()
            pred_vals = np.asarray(pred).ravel()

            # optionally convert kg/kg values to ppmv
            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                cams_vals = cams_vals * ppm_factor
                pred_vals = pred_vals * ppm_factor

            frame_data[(ch, hr)] = (lat, lon, cams_vals, pred_vals)

            all_vals = np.concatenate([cams_vals, pred_vals])
            cur_min = float(np.nanmin(all_vals))
            cur_max = float(np.nanmax(all_vals))
            if ch not in channel_ranges:
                channel_ranges[ch] = (cur_min, cur_max)
            else:
                prev_min, prev_max = channel_ranges[ch]
                channel_ranges[ch] = (min(prev_min, cur_min), max(prev_max, cur_max))

    # apply color_max_ppb cap and warn about extreme values
    for ch in list(channel_ranges):
        vmin, vmax = channel_ranges[ch]
        ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
        # warn about suspiciously large values (> 1 ppm)
        if ppm_factor is not None and vmax > 1.0:
            _logger.warning(
                f"Channel {ch}: max value {vmax:.4g} ppm (> 1 ppm) \u2013 "
                f"this may indicate a data or conversion issue."
            )

        if color_max_ppb is not None:
            if ppm_factor is not None:
                cap = color_max_ppb / 1000.0  # ppb -> ppm
            else:
                inv = _channel_ppm_factor(ch)
                cap = (color_max_ppb / 1000.0 / inv) if inv else color_max_ppb
            if vmax > cap:
                _logger.warning(
                    f"Channel {ch}: max value {vmax:.4g} exceeds color cap "
                    f"({color_max_ppb} ppb = {cap:.4g}); clipping colour scale."
                )
            vmax = min(vmax, cap)
        vmin = max(vmin, 0)  # mixing ratios are non-negative
        channel_ranges[ch] = (vmin, vmax)
        _logger.info(
            f"Value colour range for {ch}: [{vmin:.4g}, {vmax:.4g}]"
        )

    # ---- second pass: plot with fixed per-channel colour range (parallel) -
    _logger.info("Value maps [2/2]: rendering %d frames using %d threads \u2026", total_frames, _N_WORKERS)
    render_tasks = []
    for ch in channels:
        if ch not in channel_ranges:
            continue
        vmin, vmax = channel_ranges[ch]
        value_unit = "ppm" if (convert_to_ppm and _channel_ppm_factor(ch) is not None) else "kg kg\u207b\u00b9"

        for hr in forecast_hours:
            key = (ch, hr)
            if key not in frame_data:
                continue
            lat, lon, cams_vals, pred_vals = frame_data[key]
            render_tasks.append((
                ch, hr,
                np.asarray(lat).ravel(),
                np.asarray(lon).ravel(),
                np.asarray(cams_vals).ravel(),
                np.asarray(pred_vals).ravel(),
                vmin, vmax, value_unit,
                value_dir / f"values_{ch}_fstep_{hr:03d}.png",
            ))

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        futures = {
            pool.submit(_render_value_frame, *args): args[0:2]
            for args in render_tasks
        }
        for fi, future in enumerate(as_completed(futures), 1):
            ch_done, hr_done = futures[future]
            fname = future.result()
            _logger.info(
                "  Saved value frame %d/%d: %s", fi, len(render_tasks), Path(fname).name,
            )

    return value_dir


def _build_animation_from_frames(
    frame_dir: "Path",
    channels: list[str],
    forecast_hours: list[int],
    run_id: str,
    fps: float = 2.0,
    prefix: str = "values",
) -> list["Path"]:
    """Assemble per-step PNG frames into per-channel GIF animations.

    Parameters
    ----------
    frame_dir : Path
        Directory that contains the individual PNG frames.
    channels : list[str]
        Channel names to animate.
    forecast_hours : list[int]
        Forecast hours in the desired frame order.
    run_id : str
        Run identifier included in the output GIF filename.
    fps : float
        Frames per second.  Defaults to 2.
    prefix : str
        Filename prefix used when the frames were saved
        (``"values"`` or ``"bias"``).

    Returns
    -------
    list[Path]
        Paths to the GIF files that were created.
    """
    from PIL import Image

    duration_ms = int(1000 / fps) if fps > 0 else 400
    anim_dir = frame_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(
        "Assembling animations for %d channels (%d forecast hours each) …",
        len(channels), len(forecast_hours),
    )
    gif_paths: list[Path] = []
    for ci, ch in enumerate(channels, 1):
        _logger.info("  Channel %d/%d: %s", ci, len(channels), ch)
        frames: list[Image.Image] = []
        for hr in sorted(forecast_hours):
            png = frame_dir / f"{prefix}_{ch}_fstep_{hr:03d}.png"
            if png.exists():
                frames.append(Image.open(png).copy())
            else:
                _logger.warning(f"Frame {png} not found – skipping in animation.")

        if not frames:
            _logger.warning(
                f"No frames found for channel {ch} in {frame_dir} – skipping animation."
            )
            continue

        gif_path = anim_dir / f"animation_{prefix}_{ch}_{run_id}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        _logger.info(f"Saved animation: {gif_path}")
        gif_paths.append(gif_path)

    return gif_paths


def _plot_rmse_curves(
    rmse_dir: Path,
    valid_steps: list[int],
    wg_rmse: dict[str, list[float]],
    cams_rmse: dict[str, list[float]],
    run_id: str,
) -> None:
    """Write line plots of RMSE vs forecast step for each channel."""
    n_channels = len(wg_rmse)
    _logger.info("Generating RMSE curves for %d channels …", n_channels)
    for ci, (ch, wg_vals) in enumerate(wg_rmse.items(), 1):
        _logger.info("  RMSE curve %d/%d: %s", ci, n_channels, ch)
        cams_vals = cams_rmse[ch]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(valid_steps, wg_vals, marker="o", label="WeatherGen")
        ax.plot(valid_steps, cams_vals, marker="s", label="CAMS Forecast")
        ax.set_xlabel("Forecast Hour")
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE Comparison – {ch}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = rmse_dir / f"rmse_comparison_{ch}_{run_id}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"Saved RMSE comparison plot: {fname}")


def _rmse_scorecard(
    rmse_dir: Path,
    valid_steps: list[int],
    wg_rmse: dict[str, list[float]],
    cams_rmse: dict[str, list[float]],
    run_id: str,
    rel_pct_clip: float | None = 40.0,
) -> Path:
    """Create a tabular RMSE scorecard and save as CSV.

    The returned DataFrame has columns ``forecast_step, channel, wg_rmse,
    cams_rmse`` and is written to ``<rmse_dir>/rmse_scorecard_<run_id>.csv``.
    """
    records = []
    for ch in wg_rmse:
        for step, wg_val, cams_val in zip(valid_steps, wg_rmse[ch], cams_rmse[ch]):
            better = "tie"
            if wg_val < cams_val:
                better = "wg"
            elif cams_val < wg_val:
                better = "cams"
            records.append(
                {
                    "forecast_step": step,
                    "channel": ch,
                    "wg_rmse": wg_val,
                    "cams_rmse": cams_val,
                    "better": better,
                }
            )
    df = pd.DataFrame(records)
    scorecard_path = rmse_dir / f"rmse_scorecard_{run_id}.csv"
    df.to_csv(scorecard_path, index=False)
    _logger.info(f"Written RMSE scorecard to: {scorecard_path}")

    _logger.info("Rendering scorecard table image …")
    # also render the table as an image for quick visual inspection
    try:
        fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.3 + 1), dpi=300)
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        fig.tight_layout()
        img_path = rmse_dir / f"rmse_scorecard_{run_id}.png"
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"Written RMSE scorecard image to: {img_path}")
    except Exception as exc:  # pragma: no cover - optional dependency
        _logger.warning(f"Could not write scorecard image: {exc}")

    _logger.info("Rendering RMSE heatmap …")
    # extra visualization: heatmap of WG vs CAMS relative RMSE per channel/step
    try:
        df_heat = df.copy()
        rel_pct_raw = (
            (df_heat["wg_rmse"] - df_heat["cams_rmse"]) / df_heat["cams_rmse"] * 100
        )
        if rel_pct_clip is not None:
            clip_lim = abs(float(rel_pct_clip))
            if clip_lim == 0.0:
                clip_lim = 1.0
            df_heat["rel_pct"] = rel_pct_raw.clip(-clip_lim, clip_lim)
            n_clipped = int(np.sum(np.abs(rel_pct_raw.values) > clip_lim))
            if n_clipped > 0:
                _logger.info(
                    "Clipped %d RMSE relative values to +/-%.1f%% for heatmap display.",
                    n_clipped,
                    clip_lim,
                )
            lim = clip_lim
        else:
            df_heat["rel_pct"] = rel_pct_raw
            vals = df_heat["rel_pct"].values
            lim = float(np.nanmax(np.abs(vals))) if vals.size > 0 else 1.0

        channels = sorted(df_heat["channel"].unique(), key=_scorecard_channel_order)
        steps = sorted(df_heat["forecast_step"].unique())
        nchan = len(channels)
        # leave some vertical space between rows so the maps aren't squeezed;
        # we still adjust bottom margin later for the colorbar
        fig, axes = plt.subplots(
            nchan,
            1,
            figsize=(max(6, len(steps)), 1.5 * nchan),
            dpi=300,
            gridspec_kw={"hspace": 0.4},
        )
        if nchan == 1:
            axes = [axes]
        # track the image objects we create so we can always build a colorbar
        images = []
        for ax, ch in zip(axes, channels):
            vals = (
                df_heat[df_heat["channel"] == ch]
                .set_index("forecast_step")["rel_pct"]
                .reindex(steps)
                .values.reshape(1, -1)
            )
            im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
            images.append(im)
            ax.set_ylabel(ch)
            ax.set_yticks([])
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(steps)

        # use the last image created as the mappable for the shared colorbar
        mappable = images[-1] if images else None

        if mappable is not None:
            # place a single horizontal colorbar underneath all rows; increase
            # pad so it doesn’t overlap the top panel. use a small fraction so the
            # bar itself isn’t too tall.
            cbar = fig.colorbar(
                mappable,
                ax=axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes],
                orientation="horizontal",
                pad=0.15,
                fraction=0.05,
            )
            cbar.set_label("RMSE relative to CAMS (%)")

            # make room for the colorbar and avoid clipping when saving the figure
            fig.subplots_adjust(bottom=0.25)
        else:
            _logger.warning("No mappable found for heatmap, skipping colorbar")

        heat_path = rmse_dir / f"rmse_scorecard_heatmap_{run_id}.png"
        # avoid calling tight_layout after the colorbar, it often shrinks the
        # axes and pushes the bar off the figure; rely on the explicit subplots
        # adjustment instead and give a generous pad so the colorbar isn’t clipped.
        fig.savefig(heat_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        _logger.info(f"Written RMSE heatmap to: {heat_path}")
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"Could not write RMSE heatmap: {exc}")

    return scorecard_path


def plot_cams_wg_comparison(
    eval_cfg: dict,
    run_id: str,
    cams_cfg: dict,
    forecast_steps: list[int],
):
    """
    Plot global bias maps and RMSE-vs-forecast-step comparison between
    WG predictions and CAMS forecasts.

    Parameters
    ----------
    eval_cfg : dict
        Per-run evaluation configuration.
    run_id : str
        Unique identifier for the run.
    cams_cfg : dict
        Configuration for CAMS data.
    forecast_steps : list[int]
        List of forecast steps to evaluate.
    """
    stream = cams_cfg.get("cams_stream", "CAMSEAC4")
    channels = cams_cfg.get("cams_channels", None)

    # plot control flags
    plot_bias_maps_flag = cams_cfg.get("plot_bias_maps", False)
    plot_value_maps_flag = cams_cfg.get("plot_value_maps", False)
    create_video_flag = cams_cfg.get("create_video", False)
    fps = float(cams_cfg.get("fps", 2.0))
    plot_rmse_flag = cams_cfg.get("plot_rmse_curves", False)
    write_scorecard_flag = cams_cfg.get("write_scorecard", False)
    convert_to_ppm = bool(cams_cfg.get("convert_to_ppm", False))
    _raw_rmse_rel_clip = cams_cfg.get("rmse_rel_pct_clip", 40.0)
    rmse_rel_pct_clip: float | None = (
        float(_raw_rmse_rel_clip) if _raw_rmse_rel_clip is not None else None
    )
    # colour scale cap in ppb – keeps animations readable and flags bad values
    _raw_cap = cams_cfg.get("color_max_ppb", None)
    color_max_ppb: float | None = float(_raw_cap) if _raw_cap is not None else None

    # --- readers ---------------------------------------------------------
    wg_reader = WeatherGenZarrReader(eval_cfg, run_id)
    # step_hrs may not be present in the per-run inference config; allow
    # it to be specified in cams_cfg (the top-level evaluation section).
    if "step_hrs" in cams_cfg:
        wg_reader.step_hrs = int(cams_cfg["step_hrs"])
    cams_reader = CAMSForecastReader(cams_cfg)

    # handle user specification of "all" or other shorthand for steps
    if isinstance(forecast_steps, str) and forecast_steps.lower() == "all":
        step_hrs = wg_reader.step_hrs
        wg_hours = set(int(idx) * step_hrs for idx in wg_reader.get_forecast_steps())
        cams_hours = _cams_steps_as_hours(cams_reader)
        forecast_steps = sorted(wg_hours | cams_hours)
    elif isinstance(forecast_steps, (int, float)):
        forecast_steps = [int(forecast_steps)]
    else:
        forecast_steps = list(forecast_steps)

    # align hours and get mapping to raw WG values
    forecast_hours, wg_map = _align_forecast_steps(wg_reader, cams_reader, forecast_steps)
    if not forecast_hours:
        _logger.error("No common forecast steps between WG and CAMS – aborting comparison.")
        return

    # create list of raw WG steps corresponding to hours for data retrieval
    wg_raw_steps = [wg_map[h] for h in forecast_hours]

    # replace forecast_steps variable with hours for looping convenience
    forecast_steps = forecast_hours

    # Determine channels to compare – prefer the per-run stream config
    if channels is None:
        stream_cfg = eval_cfg.get("streams", {}).get(stream, {})
        channels = stream_cfg.get("channels", None)
    if channels is None:
        channels = wg_reader.get_channels(stream)

    # Keep only channels available in the selected CAMS forecast dataset.
    missing_channels = [ch for ch in channels if not cams_reader.supports_channel(ch)]
    if missing_channels:
        _logger.warning(
            "Skipping channels not found in CAMS dataset %s: %s",
            cams_reader.cams_forecast_path,
            missing_channels,
        )
    channels = [ch for ch in channels if cams_reader.supports_channel(ch)]
    if not channels:
        _logger.error(
            "No requested channels are available in CAMS dataset %s. "
            "Set evaluation.cams_forecast_filename to a file containing your channels.",
            cams_reader.cams_forecast_path,
        )
        return

    # --- plotter for bias maps ------------------------------------------
    plotter_cfg = {
        "image_format": "png",
        "dpi_val": 300,
        "fig_size": (10, 6),
        "regions": ["global"],
        "fps": fps,
    }
    # The eval configuration may already include the run_id in the
    # directory path (e.g. results_base_dir or runplot_base_dir set to
    # "results/<run_id>").  The Plotter class expects the path it receives
    # to *already* include the run id (see its docstring).  In the past we
    # always appended ``/ run_id`` which resulted in nested directories
    # like ``results/foo/foo`` when the base was ``results/foo``.
    base_dir = Path(eval_cfg.get("runplot_base_dir", eval_cfg.get("results_base_dir", ".")))
    if base_dir.name == run_id:
        output_dir = base_dir
    else:
        output_dir = base_dir / run_id

    plotter = Plotter(plotter_cfg, output_dir, stream=stream)

    # --- load WG data once (all requested forecast steps) ---------------
    _logger.info(
        "Loading WG data: stream=%s, %d forecast steps, %d channels",
        stream, len(wg_raw_steps), len(channels),
    )
    wg_data = wg_reader.get_data(stream=stream, fsteps=wg_raw_steps, channels=channels)

    # Force eager computation of any dask-backed arrays so that subsequent
    # `.values` / `.sel()` calls don't each trigger a separate dask compute.
    for fstep_key in list(wg_data.target):
        wg_data.target[fstep_key] = wg_data.target[fstep_key].load()
        wg_data.prediction[fstep_key] = wg_data.prediction[fstep_key].load()
    _logger.info("WG data loaded into memory.")
    wg_rmse_per_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    cams_rmse_per_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    valid_fsteps: list[int] = []
    # Cache CAMS data keyed by (channel, hour) so that bias/value map
    # functions do not have to re-read from zarr.
    cams_cache: dict[tuple[str, int], np.ndarray] = {}

    # Phase 1: verify timestep alignment for all hours (fast, sequential)
    _logger.info("Verifying timestep alignment for %d forecast hours \u2026", len(forecast_steps))
    _hr_info: dict[int, tuple] = {}  # hr -> (raw, wg_target, wg_pred, init_time)
    for hr in forecast_steps:
        raw = wg_map[hr]
        if raw not in wg_data.target:
            _logger.warning(f"Forecast step {hr}h (raw {raw}) not found in WG output \u2013 skipping.")
            continue
        wg_target = wg_data.target[raw]
        wg_pred = wg_data.prediction[raw]
        valid_time = _verify_timestep_alignment(wg_target, wg_pred, cams_reader, hr)
        init_time = valid_time - np.timedelta64(int(hr), "h")
        _hr_info[hr] = (raw, wg_target, wg_pred, init_time)
        valid_fsteps.append(hr)
        _logger.info(
            "  Forecast hour %dh: verified alignment at %s",
            hr, _format_datetime64(valid_time),
        )

    # Phase 2: pre-load CAMS data and compute RMSE
    # Batch-load all CAMS data into memory and interpolate to WG grid in
    # one vectorised call per forecast hour.  This eliminates the dask/zarr
    # contention that made the old per-(channel, hour) ThreadPoolExecutor
    # approach extremely slow.
    cams_cache = _preload_and_interpolate_cams(
        cams_reader, channels, valid_fsteps, _hr_info,
    )

    # Pre-load CAMS analysis for ground-truth RMSE computation.
    # When a ``cams_analysis_filename`` is configured, the CAMS forecast
    # RMSE is computed on the native CAMS grid against the operational
    # CAMS analysis (matching the standalone scorecard script).  Otherwise
    # we fall back to RMSE(CAMS_forecast, WG_target) on the WG scatter grid.
    cams_native_rmse: dict[tuple[str, int], float] | None = None
    cams_analysis_filename = cams_cfg.get("cams_analysis_filename")
    if cams_analysis_filename:
        cams_analysis_path = Path(cams_cfg.get("cams_base_dir")) / cams_analysis_filename
        if cams_analysis_path.exists():
            cams_native_rmse = _compute_cams_native_rmse(
                cams_reader, cams_analysis_path, channels, valid_fsteps, _hr_info,
            )
        else:
            _logger.warning(
                "CAMS analysis file %s not found; falling back to WG targets for CAMS RMSE.",
                cams_analysis_path,
            )

    _logger.info(
        "Computing RMSE for %d forecast hours × %d channels …",
        len(valid_fsteps), len(channels),
    )
    for hr in valid_fsteps:
        _raw, wg_target, wg_pred, _init_time = _hr_info[hr]
        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)

            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                pred_ppm = pred * ppm_factor
                tar_ppm = tar * ppm_factor
            else:
                pred_ppm, tar_ppm = pred, tar

            wg_rmse_per_channel[ch].append(float(_compute_rmse(pred_ppm, tar_ppm)))

            # CAMS RMSE: prefer native-grid computation when available
            if cams_native_rmse is not None and (ch, hr) in cams_native_rmse:
                cams_rmse_val = cams_native_rmse[(ch, hr)]
                if ppm_factor is not None:
                    cams_rmse_val *= ppm_factor
                cams_rmse_per_channel[ch].append(cams_rmse_val)
            else:
                cams_vals = cams_cache[(ch, hr)]
                tar_flat = np.asarray(tar.values).ravel()
                if ppm_factor is not None:
                    cams_vals_rmse = cams_vals * ppm_factor
                    tar_flat_rmse = tar_flat * ppm_factor
                else:
                    cams_vals_rmse, tar_flat_rmse = cams_vals, tar_flat
                cams_rmse_per_channel[ch].append(
                    float(np.sqrt(((cams_vals_rmse - tar_flat_rmse) ** 2).mean()))
                )
    _logger.info("RMSE computation complete.")

    # ---- Post-RMSE verification: flag suspicious RMSE patterns -----------
    _logger.info("Verifying RMSE results …")
    for ch in channels:
        wg_vals = wg_rmse_per_channel[ch]
        cams_vals = cams_rmse_per_channel[ch]
        if len(cams_vals) > 1:
            cams_arr = np.array(cams_vals)
            cams_cv = float(np.std(cams_arr) / np.mean(cams_arr)) if np.mean(cams_arr) > 0 else 0.0
            if cams_cv < 0.01:
                _logger.warning(
                    "SUSPICIOUS RMSE: CAMS %s RMSE is nearly constant across %d "
                    "forecast steps (CV=%.4f). Values: %s. "
                    "This may indicate the same CAMS data is being used for every step.",
                    ch, len(cams_vals), cams_cv,
                    [f"{v:.4e}" for v in cams_vals],
                )
        for i, hr in enumerate(valid_fsteps):
            if i < len(wg_vals) and i < len(cams_vals):
                wg_v, cams_v = wg_vals[i], cams_vals[i]
                if cams_v > 0 and wg_v > 0:
                    ratio = wg_v / cams_v
                    if ratio > 10 or ratio < 0.1:
                        _logger.warning(
                            "LARGE RMSE ratio for %s at %dh: WG=%.4e vs CAMS=%.4e "
                            "(ratio=%.2g). Check data selection and units.",
                            ch, hr, wg_v, cams_v, ratio,
                        )
    _logger.info("RMSE verification complete.")

    # Once the iteration over forecast hours and channels is complete we
    # optionally create outputs based on the flags supplied in the CAMS
    # configuration.
    if plot_bias_maps_flag:
        _logger.info("Generating bias maps for common forecast steps")
        bias_dir = plotter.out_plot_basedir / stream / "maps" / "bias_compare"
        _plot_bias_maps(
            plotter,
            wg_reader,
            cams_reader,
            stream,
            channels,
            forecast_steps,
            wg_map,
            run_id,
            convert_to_ppm=convert_to_ppm,
            color_max_ppb=color_max_ppb,
            wg_data=wg_data,
            cams_cache=cams_cache,
        )
        if create_video_flag:
            _logger.info("Building bias map animations")
            _build_animation_from_frames(
                bias_dir,
                channels,
                forecast_steps,
                run_id,
                fps=fps,
                prefix="bias",
            )

    if plot_value_maps_flag:
        _logger.info("Generating value comparison maps (CAMS vs WG)")
        value_dir = _plot_value_maps(
            plotter,
            wg_reader,
            cams_reader,
            stream,
            channels,
            forecast_steps,
            wg_map,
            run_id,
            convert_to_ppm=convert_to_ppm,
            color_max_ppb=color_max_ppb,
            wg_data=wg_data,
            cams_cache=cams_cache,
        )
        if create_video_flag:
            _logger.info("Building value map animations")
            _build_animation_from_frames(
                value_dir,
                channels,
                forecast_steps,
                run_id,
                fps=fps,
                prefix="values",
            )

    if plot_rmse_flag or write_scorecard_flag:
        rmse_dir = output_dir / stream / "rmse"
        rmse_dir.mkdir(parents=True, exist_ok=True)

        if plot_rmse_flag:
            _plot_rmse_curves(
                rmse_dir,
                valid_fsteps,
                wg_rmse_per_channel,
                cams_rmse_per_channel,
                run_id,
            )

        if write_scorecard_flag:
            _rmse_scorecard(
                rmse_dir,
                valid_fsteps,
                wg_rmse_per_channel,
                cams_rmse_per_channel,
                run_id,
                rel_pct_clip=rmse_rel_pct_clip,
            )


