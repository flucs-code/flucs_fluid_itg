"""Plot time-averaged zonal profiles for the cold-ion ITG system."""

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
from flucs.postprocessing import FlucsPostProcessing
from flucs_fluid_itg.cold_itg_2d_fourier.profile_postprocessing import (
    load_profile_averages,
    parse_time_window,
    spectral_derivative,
)


def plot_zonal_profiles(
    post,
    *,
    groups=None,
    time=None,
    fraction=None,
):
    """Create one five-panel zonal-profile figure per simulation."""

    nc_paths = post.get_valid_netcdf_paths("zonal_profiles/phi")
    for nc_path in nc_paths:
        average = load_profile_averages(
            post,
            nc_path,
            {
                "phi": "zonal_profiles/phi",
                "T": "zonal_profiles/T",
            },
            groups=groups,
            time_min=None if time is None else time[0],
            time_max=None if time is None else time[1],
            fraction=fraction,
        )
        x = average.x
        phi = average.values["phi"]
        temperature = average.values["T"]
        flow = spectral_derivative(phi, x)
        shear = spectral_derivative(phi, x, order=2)
        temperature_gradient = spectral_derivative(temperature, x)

        fig, axes = plt.subplots(
            5, 1, sharex=True, layout="constrained", figsize=(7, 11)
        )
        run_name = pl.Path(nc_path).parent.name
        figure_name = f"zonal_profiles_{run_name}"
        fig.canvas.manager.set_window_title(figure_name)
        fig.suptitle(
            rf"{run_name}: $t\in[{average.time_min:.4g},"
            rf"{average.time_max:.4g}]$"
        )

        axes[0].plot(x, phi - phi.mean(), color="black")
        axes[0].set_ylabel(r"$\overline{\phi}-\langle\overline{\phi}\rangle_x$")

        axes[1].plot(x, flow, color="tab:blue")
        axes[1].set_ylabel(r"$u_y=\partial_x\overline{\phi}$")

        axes[2].plot(x, shear, color="tab:orange")
        axes[2].axhline(
            average.gamma_max, color="black", linestyle="dotted"
        )
        axes[2].axhline(
            -average.gamma_max, color="black", linestyle="dotted"
        )
        axes[2].set_ylabel(r"$S=\partial_x^2\overline{\phi}$")

        axes[3].plot(x, temperature, color="tab:red")
        axes[3].set_ylabel(r"$\overline{T}$")

        axes[4].plot(x, temperature_gradient, color="tab:green")
        axes[4].axhline(
            average.parameters["kappaT"],
            color="black",
            linestyle="dotted",
        )
        axes[4].set_ylabel(r"$\partial_x\overline{T}$")
        axes[4].set_xlabel(r"$x$")

        for axis in axes:
            axis.axhline(0.0, color="0.7", linewidth=0.7)
            axis.set_xlim(x[0], x[-1])

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
        description="Plot time-averaged zonal profiles.",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--time",
        type=parse_time_window,
        default=None,
        metavar="START:END",
        help="Average over the inclusive START:END time window.",
    )
    window.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Average this final fraction of the available time span.",
    )
    args = parser.parse_args()

    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.1d.nc"],
        constraint="both",
    )
    plot_zonal_profiles(
        post,
        groups=args.groups,
        time=args.time,
        fraction=args.fraction,
    )
