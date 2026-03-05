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
    convert_to_ppm: bool = False,
) -> None:
    """Produce and save global bias scatter maps for each forecast step/channel.

    Maps are written under ``plotter.out_plot_basedir/<stream>/maps/{wg_bias,cams_bias}``.
    """
    # make sure output directories exist before looping; the Plotter
    # does not create them automatically and missing dirs were causing
    # errors during evaluation (see bug report).
    for tag in ("wg_bias", "cams_bias"):
        outdir = plotter.get_map_output_dir(tag)
        if not outdir.exists():
            _logger.info(f"Creating directory {outdir}")
            outdir.mkdir(parents=True, exist_ok=True)

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
            # extract coordinates once
            lat = tar["lat"].values
            lon = tar["lon"].values
            wg_bias = pred - tar

            _logger.info(f"Processing forecast step {hr}h, channel {ch}")
            # Ensure wg_bias is calculated before logging
            wg_bias = pred - tar

            # Ensure cams_vals is calculated before logging
            cams_vals = cams_reader.get_data(ch, step=hr, target_lat=lat, target_lon=lon)
            cams_vals = np.asarray(cams_vals).ravel()

            # Log detailed shapes for debugging
            _logger.info(f"wg_bias shape: {wg_bias.shape}")
            _logger.info(f"cams_vals shape: {cams_vals.shape}")
            wg_bias_da, cams_bias_da = _prepare_bias_da(pred, tar, cams_vals)

            # optionally convert kg/kg bias to ppmv
            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                wg_bias_da = wg_bias_da * ppm_factor
                cams_bias_da = cams_bias_da * ppm_factor
                _logger.info(f"Applied ppm conversion factor {ppm_factor:.4g} to {ch} bias")

            plotter.update_data_selection(
                {"sample": 0, "stream": stream, "forecast_step": hr}
            )
            # compute symmetric color bounds so zero is centered on the bar
            all_vals = np.concatenate([
                np.asarray(wg_bias_da).ravel(),
                np.asarray(cams_bias_da).ravel(),
            ])
            if all_vals.size > 0:
                maxabs = float(np.nanmax(np.abs(all_vals)))
            else:
                maxabs = 0.0
            map_opts = {"vmin": -maxabs, "vmax": maxabs, "colormap": "coolwarm"}

            # produce a combined figure with two side-by-side maps sharing
            # the same symmetric colourbar centred on zero
            import cartopy.crs as ccrs

            # directory for combined bias plots
            combo_dir = plotter.out_plot_basedir / stream / "maps" / "bias_compare"
            combo_dir.mkdir(parents=True, exist_ok=True)

            fig = plt.figure(figsize=(16, 8), dpi=300)
            axs = [
                fig.add_subplot(1, 2, 1, projection=ccrs.Robinson()),
                fig.add_subplot(1, 2, 2, projection=ccrs.Robinson()),
            ]
            titles = [
                "CAMS Forecast – Analysis",
                "WG Prediction – Target",
            ]
            das = [cams_bias_da, wg_bias_da]

            for ax, da, title in zip(axs, das, titles):
                ax.coastlines()
                scatter_plt = ax.scatter(
                    da["lon"],
                    da["lat"],
                    c=da.values,
                    cmap=map_opts.get("colormap", "coolwarm"),
                    vmin=map_opts.get("vmin"),
                    vmax=map_opts.get("vmax"),
                    transform=ccrs.PlateCarree(),
                    s=1,
                    linewidths=0.0,
                )
                ax.set_global()
                ax.set_title(f"{title} – {ch} (fstep {hr})")

            # shared colorbar beneath both axes
            mappable = scatter_plt
            cbar = fig.colorbar(
                mappable,
                ax=axs,
                orientation="horizontal",
                pad=0.02,                # smaller pad to keep colorbar close to axes
                fraction=0.04,           # make bar thinner
            )
            bias_unit = "ppm" if (convert_to_ppm and _channel_ppm_factor(ch) is not None) else "kg kg\u207b\u00b9"
            cbar.set_label(f"Bias ({bias_unit})")

            # explicitly adjust margins so the colorbar isn't clipped; we want
            # some space at bottom for ticks and label
            fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

            fname = combo_dir / f"bias_{ch}_fstep_{hr:03d}.png"
            fig.savefig(fname, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            plotter.clean_data_selection()


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
) -> "Path":
    """Produce and save side-by-side global maps of CAMS forecast vs WG prediction.

    Each frame is a Robinson-projection scatter plot with two panels:
    *CAMS Forecast* (left) and *WG Prediction* (right), sharing a common
    colour scale derived from the combined range of both datasets.

    Frames are written under
    ``plotter.out_plot_basedir/<stream>/maps/value_compare/``
    with the naming convention ``values_{ch}_fstep_{hr:03d}.png``.

    Returns
    -------
    Path
        Output directory where the frames were saved.
    """
    import cartopy.crs as ccrs

    value_dir = plotter.out_plot_basedir / stream / "maps" / "value_compare"
    value_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Saving value comparison maps to {value_dir}")

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
                _logger.info(f"Applied ppm conversion factor {ppm_factor:.4g} to {ch} values")

            # shared colour range so both panels are directly comparable
            all_vals = np.concatenate([cams_vals, pred_vals])
            vmin = float(np.nanmin(all_vals))
            vmax = float(np.nanmax(all_vals))

            _logger.info(
                f"Value map – fstep {hr}h, channel {ch}: "
                f"vmin={vmin:.4g}, vmax={vmax:.4g}"
            )

            fig = plt.figure(figsize=(16, 8), dpi=300)
            axs = [
                fig.add_subplot(1, 2, 1, projection=ccrs.Robinson()),
                fig.add_subplot(1, 2, 2, projection=ccrs.Robinson()),
            ]
            titles = ["CAMS Forecast", "WG Prediction"]
            data_pairs = [
                (lon, lat, cams_vals),
                (lon, lat, pred_vals),
            ]

            last_sc = None
            for ax, title, (lons, lats, vals) in zip(axs, titles, data_pairs):
                ax.coastlines()
                last_sc = ax.scatter(
                    lons,
                    lats,
                    c=vals,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    s=1,
                    linewidths=0.0,
                )
                ax.set_global()
                ax.set_title(f"{title} – {ch} (fstep {hr}h)")

            cbar = fig.colorbar(
                last_sc,
                ax=axs,
                orientation="horizontal",
                pad=0.02,
                fraction=0.04,
            )
            value_unit = "ppm" if (convert_to_ppm and _channel_ppm_factor(ch) is not None) else "kg kg\u207b\u00b9"
            cbar.set_label(f"{ch} ({value_unit})")

            fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

            fname = value_dir / f"values_{ch}_fstep_{hr:03d}.png"
            fig.savefig(fname, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            _logger.info(f"Saved value comparison map: {fname}")

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

    gif_paths: list[Path] = []
    for ch in channels:
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

    # extra visualization: heatmap of WG vs CAMS relative RMSE per channel/step
    try:
        df_heat = df.copy()
        df_heat["rel_pct"] = (
            (df_heat["wg_rmse"] - df_heat["cams_rmse"]) / df_heat["cams_rmse"] * 100
        )
        channels = sorted(df_heat["channel"].unique())
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
        # determine symmetric color limits from the data so the colormap isn't
        # overly compressed. fall back to 1.0 if the dataframe is empty.
        all_rel = df_heat["rel_pct"].values
        if all_rel.size > 0:
            lim = float(np.nanmax(np.abs(all_rel)))
        else:
            lim = 1.0

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
    # hook for further settings if needed
    # bias_map_opts = cams_cfg.get("bias_map_opts", {})
    # rmse_plot_opts = cams_cfg.get("rmse_plot_opts", {})
    # scorecard_opts = cams_cfg.get("scorecard_opts", {})

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

            # fetch CAMS values and compute RMSE; optionally convert to ppm first
            cams_vals = cams_reader.get_data(ch, step=hr, target_lat=tar["lat"].values, target_lon=tar["lon"].values)
            cams_vals = np.asarray(cams_vals).ravel()
            tar_flat = np.asarray(tar.values).ravel()

            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                pred_ppm = pred * ppm_factor
                tar_ppm = tar * ppm_factor
                cams_vals_rmse = cams_vals * ppm_factor
                tar_flat_rmse = tar_flat * ppm_factor
            else:
                pred_ppm, tar_ppm = pred, tar
                cams_vals_rmse, tar_flat_rmse = cams_vals, tar_flat

            wg_rmse_per_channel[ch].append(float(_compute_rmse(pred_ppm, tar_ppm)))
            cams_rmse_per_channel[ch].append(
                float(np.sqrt(((cams_vals_rmse - tar_flat_rmse) ** 2).mean()))
            )

            # we simply accumulate RMSE values here; detailed bias maps
            # are produced later by _plot_bias_maps if requested.

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
            )


