# (C) Copyright 2025 WeatherGenerator contributors.
# Licensed under Apache 2.0.

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from weathergen.common.io import CAMSForecastReader
from weathergen.evaluate.io.wegen_reader import WeatherGenZarrReader
from weathergen.evaluate.plotting.plotter import Plotter

_logger = logging.getLogger(__name__)


def _compute_rmse(pred: xr.DataArray, target: xr.DataArray, dim: str = "ipoint") -> xr.DataArray:
    """Compute root-mean-square error along *dim*."""
    return np.sqrt(((pred - target) ** 2).mean(dim=dim))



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



def _plot_bias_maps(
    plotter: Plotter,
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    stream: str,
    channels: list[str],
    forecast_hours: list[int],
    wg_map: dict[int, int],
    run_id: str,
) -> None:
    """Produce and save global bias scatter maps for each forecast step/channel.

    Maps are written under ``plotter.out_plot_basedir/<stream>/maps/{wg_bias,cams_bias}``.
    """
    for hr in forecast_hours:
        raw = wg_map[hr]
        wg_data = wg_reader.get_data(stream=stream, fsteps=[raw], channels=channels)
        if raw not in wg_data.target:
            _logger.warning(
                f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping."
            )
            continue
        wg_target = wg_data.target[raw]
        wg_pred = wg_data.prediction[raw]

        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)
            lat = tar["lat"].values
            lon = tar["lon"].values
            wg_bias = pred - tar
            lat = tar["lat"].values
            lon = tar["lon"].values

            _logger.info(f"Processing forecast step {hr}h, channel {ch}")
            # Ensure wg_bias is calculated before logging
            wg_bias = pred - tar

            # Ensure cams_vals is calculated before logging
            cams_vals = cams_reader.get_data(ch, step=hr, target_lat=lat, target_lon=lon)
            cams_vals = np.asarray(cams_vals).ravel()


# Log detailed shapes for debugging
            wg_bias_da, cams_bias_da = _prepare_bias_da(pred, tar, cams_vals)
            plotter.update_data_selection(
                {"sample": 0, "stream": stream, "forecast_step": hr}
            )
            plotter.scatter_plot(
                wg_bias_da,
                plotter.get_map_output_dir("wg_bias"),
                f"wg_bias_{ch}",
                "global",
                title=f"WG Bias – {ch} (fstep {fstep})",
            )
            plotter.scatter_plot(
                cams_bias_da,
                plotter.get_map_output_dir("cams_bias"),
                f"cams_bias_{ch}",
                "global",
                title=f"CAMS Bias – {ch} (fstep {fstep})",
            )
            plotter.clean_data_selection()


def _plot_rmse_curves(
    rmse_dir: Path,
    valid_steps: list[int],
    wg_rmse: dict[str, list[float]],
    cams_rmse: dict[str, list[float]],
    run_id: str,
) -> None:
    """Write line plots of RMSE vs forecast step for each channel."""
    for ch, wg_vals in wg_rmse.items():
        cams_vals = cams_rmse[ch]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(valid_steps, wg_vals, marker="o", label="WeatherGen")
        ax.plot(valid_steps, cams_vals, marker="s", label="CAMS Forecast")
        ax.set_xlabel("Forecast Step")
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
) -> Path:
    """Create a tabular RMSE scorecard and save as CSV.

    The returned DataFrame has columns ``forecast_step, channel, wg_rmse,
    cams_rmse`` and is written to ``<rmse_dir>/rmse_scorecard_<run_id>.csv``.
    """
    records = []
    for ch in wg_rmse:
        for step, wg_val, cams_val in zip(valid_steps, wg_rmse[ch], cams_rmse[ch]):
            records.append(
                {
                    "forecast_step": step,
                    "channel": ch,
                    "wg_rmse": wg_val,
                    "cams_rmse": cams_val,
                }
            )
    df = pd.DataFrame(records)
    scorecard_path = rmse_dir / f"rmse_scorecard_{run_id}.csv"
    df.to_csv(scorecard_path, index=False)
    _logger.info(f"Written RMSE scorecard to: {scorecard_path}")
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
    plot_rmse_flag = cams_cfg.get("plot_rmse_curves", False)
    write_scorecard_flag = cams_cfg.get("write_scorecard", False)
    # hook for further settings if needed
    # bias_map_opts = cams_cfg.get("bias_map_opts", {})
    # rmse_plot_opts = cams_cfg.get("rmse_plot_opts", {})
    # scorecard_opts = cams_cfg.get("scorecard_opts", {})

    # --- readers ---------------------------------------------------------
    wg_reader = WeatherGenZarrReader(eval_cfg, run_id)
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

    # --- plotter for bias maps ------------------------------------------
    plotter_cfg = {
        "image_format": "png",
        "dpi_val": 300,
        "fig_size": (10, 6),
        "regions": ["global"],
    }
    output_dir = Path(
        eval_cfg.get("runplot_base_dir", eval_cfg.get("results_base_dir", "."))
    ) / run_id
    plotter = Plotter(plotter_cfg, output_dir, stream=stream)

    # --- load WG data once (all requested forecast steps) ---------------
    wg_data = wg_reader.get_data(stream=stream, fsteps=wg_raw_steps, channels=channels)

    # accumulate RMSE values while iterating; maps are generated separately
    wg_rmse_per_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    cams_rmse_per_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    valid_fsteps: list[int] = []

    for hr in forecast_steps:  # hours now
        raw = wg_map[hr]
        if raw not in wg_data.target:
            _logger.warning(f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping.")
            continue

        valid_fsteps.append(hr)
        wg_target = wg_data.target[raw]      # xr.Dataset  (channel, ipoint)
        wg_pred = wg_data.prediction[raw]     # xr.Dataset  (channel, ipoint)

        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)
            lat = tar["lat"].values
            lon = tar["lon"].values
            wg_bias = pred - tar

            wg_rmse_per_channel[ch].append(float(_compute_rmse(pred, tar)))

            # Ensure cams_vals is calculated before logging
            cams_vals = cams_reader.get_data(ch, step=hr, target_lat=tar["lat"].values, target_lon=tar["lon"].values)
            cams_vals = np.asarray(cams_vals).ravel()


# # Log detailed shapes for debugging
#             wg_bias_da, cams_bias_da = _prepare_bias_da(pred, tar, cams_vals)
#             plotter.update_data_selection(
#                 {"sample": 0, "stream": stream, "forecast_step": hr}
#             )
#             plotter.scatter_plot(
#                 wg_bias_da,
#                 plotter.get_map_output_dir("wg_bias"),
#                 f"wg_bias_{ch}",
#                 "global",
#                 title=f"WG Bias – {ch} (fstep {fstep})",
#             )
#             plotter.scatter_plot(
#                 cams_bias_da,
#                 plotter.get_map_output_dir("cams_bias"),
#                 f"cams_bias_{ch}",
#                 "global",
#                 title=f"CAMS Bias – {ch} (fstep {fstep})",
#             )
#             plotter.clean_data_selection()


