"""Plot heat flux and the time evolution of the zonal flow."""

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from flucs.postprocessing import FlucsPostProcessing
from flucs_fluid_itg.cold_itg_2d_fourier.profile_postprocessing import (
    parse_time_window,
    spectral_derivative,
)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _group_ids(post, heatflux_path, profile_path, groups):
    heatflux_groups = set(
        post.get_netcdf_variables(heatflux_path)["heatflux/heatflux"]
    )
    profile_groups = set(
        post.get_netcdf_variables(profile_path)["zonal_profiles/phi"]
    )

    if groups is None:
        selected = sorted(heatflux_groups & profile_groups)
    else:
        try:
            selected = [int(group) for group in groups]
        except ValueError:
            raise ValueError("Output groups must be integer identifiers.") from None
        missing = [
            group
            for group in selected
            if group not in heatflux_groups or group not in profile_groups
        ]
        if missing:
            raise ValueError(
                "Selected output groups are not available in both output "
                f"files: {missing}."
            )

    if not selected:
        raise ValueError("No common output groups contain both diagnostics.")
    return [str(group) for group in selected]


def _validate_times(times, name):
    previous_end = None
    for time in times:
        if time.ndim != 1 or not time.size:
            raise ValueError(f"{name} time coordinates must be non-empty and 1D.")
        if not np.all(np.isfinite(time)):
            raise ValueError(f"{name} time coordinates must be finite.")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError(f"{name} times must increase within each group.")
        if previous_end is not None and time[0] < previous_end and not np.isclose(
            time[0], previous_end, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(f"{name} output groups overlap in time.")
        previous_end = time[-1]


def _load_segments(post, heatflux_path, profile_path, groups):
    heatflux_times = post.load_netcdf_variable(
        heatflux_path, "time", groups=groups, concatenate=False
    )[0]
    heatflux = post.load_netcdf_variable(
        heatflux_path,
        "heatflux/heatflux",
        groups=groups,
        concatenate=False,
    )[0]
    profile_times = post.load_netcdf_variable(
        profile_path, "time", groups=groups, concatenate=False
    )[0]
    phi, _, dimensions = post.load_netcdf_variable(
        profile_path,
        "zonal_profiles/phi",
        groups=groups,
        concatenate=False,
    )

    heatflux_times = [np.asarray(values, dtype=float) for values in heatflux_times]
    profile_times = [np.asarray(values, dtype=float) for values in profile_times]
    heatflux = [np.asarray(values) for values in heatflux]
    phi = [np.asarray(values) for values in phi]
    _validate_times(heatflux_times, "Heat-flux")
    _validate_times(profile_times, "Zonal-profile")

    x_grids = [np.asarray(dims["x"], dtype=float) for dims in dimensions]
    x = x_grids[0]
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("The x grid must contain at least two finite points.")

    for index, (time, values) in enumerate(zip(heatflux_times, heatflux)):
        if values.shape != time.shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Invalid heat-flux values in selected group {groups[index]}."
            )
    for index, (time, values, group_x) in enumerate(
        zip(profile_times, phi, x_grids)
    ):
        if values.shape != (time.size, x.size) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Invalid zonal profiles in selected group {groups[index]}."
            )
        if group_x.shape != x.shape or not np.allclose(
            group_x, x, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("The x grid changes between selected groups.")

    return heatflux_times, heatflux, profile_times, phi, x


def _time_bounds(heatflux_times, profile_times, requested):
    heatflux_min = min(time[0] for time in heatflux_times)
    heatflux_max = max(time[-1] for time in heatflux_times)
    profile_min = min(time[0] for time in profile_times)
    profile_max = max(time[-1] for time in profile_times)
    lower = max(heatflux_min, profile_min)
    upper = min(heatflux_max, profile_max)
    if requested is not None:
        lower = max(lower, requested[0])
        upper = min(upper, requested[1])
    if upper <= lower:
        raise ValueError("The selected diagnostics have no common time interval.")
    return float(lower), float(upper)


def plot_heatflux_and_zonal_flow(post, *, groups=None, time=None):
    """Create one heat-flux and zonal-flow evolution figure per simulation."""

    heatflux_paths = {
        path.parent: path
        for path in post.get_valid_netcdf_paths("heatflux/heatflux")
    }
    profile_paths = {
        path.parent: path
        for path in post.get_valid_netcdf_paths("zonal_profiles/phi")
    }
    run_paths = sorted(heatflux_paths.keys() & profile_paths.keys())
    if not run_paths:
        raise ValueError("No simulation contains both required diagnostics.")

    for run_path in run_paths:
        selected_groups = _group_ids(
            post,
            heatflux_paths[run_path],
            profile_paths[run_path],
            groups,
        )
        (
            heatflux_times,
            heatflux,
            profile_times,
            phi,
            x,
        ) = _load_segments(
            post,
            heatflux_paths[run_path],
            profile_paths[run_path],
            selected_groups,
        )
        time_min, time_max = _time_bounds(
            heatflux_times, profile_times, time
        )

        heatflux_segments = []
        for segment_time, segment_values in zip(heatflux_times, heatflux):
            selected = (segment_time >= time_min) & (segment_time <= time_max)
            if np.any(selected):
                heatflux_segments.append(
                    (segment_time[selected], segment_values[selected])
                )

        flow_segments = []
        for segment_time, segment_phi in zip(profile_times, phi):
            selected = (segment_time >= time_min) & (segment_time <= time_max)
            if np.count_nonzero(selected) >= 2:
                flow_segments.append(
                    (
                        segment_time[selected],
                        spectral_derivative(segment_phi[selected], x),
                    )
                )
        if not flow_segments:
            raise ValueError(
                "The selected interval needs at least two zonal-profile "
                "samples in one output group."
            )

        color_limit = max(
            float(np.max(np.abs(values))) for _, values in flow_segments
        )
        if color_limit == 0.0:
            color_limit = np.finfo(float).eps
        levels = np.linspace(-color_limit, color_limit, 101)
        norm = Normalize(vmin=-color_limit, vmax=color_limit)

        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            layout="constrained",
            figsize=(8, 7),
            height_ratios=(1, 2),
        )
        run_name = pl.Path(run_path).name
        figure_name = f"heatflux_zonal_flow_{run_name}"
        fig.canvas.manager.set_window_title(figure_name)
        fig.suptitle(
            rf"{run_name}: $t\in[{time_min:.4g},{time_max:.4g}]$"
        )

        for segment_time, segment_values in heatflux_segments:
            axes[0].plot(segment_time, segment_values, color="black")
        axes[0].set_ylabel(r"$Q_i/[4 n_e T_e c_s (\rho_s/L_B)^2]$")

        for segment_time, segment_flow in flow_segments:
            axes[1].contourf(
                segment_time,
                x,
                segment_flow.T,
                levels=levels,
                cmap="seismic",
                norm=norm,
            )
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap="seismic"), ax=axes[1]
        )
        colorbar.set_label(r"$u_y=\partial_x\overline{\phi}$")
        axes[1].set_ylabel(r"$x$")
        axes[1].set_xlabel(r"$(2c_s/L_B)t$")
        axes[1].set_xlim(time_min, time_max)

        post.save(
            fig,
            name=figure_name,
            suffix="png",
            save_kwargs={"dpi": 300},
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description="Plot heat flux and the time evolution of zonal flow.",
    )
    parser.add_argument(
        "--time",
        type=parse_time_window,
        default=None,
        metavar="START:END",
        help="Plot the inclusive START:END time interval.",
    )
    args = parser.parse_args()

    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.0d.nc", "output.1d.nc"],
        constraint="both",
    )
    plot_heatflux_and_zonal_flow(
        post,
        groups=args.groups,
        time=args.time,
    )
