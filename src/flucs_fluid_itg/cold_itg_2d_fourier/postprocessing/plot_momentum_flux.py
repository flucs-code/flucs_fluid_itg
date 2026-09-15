"""Plot time-averaged zonal momentum-flux profiles."""

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from flucs.postprocessing import FlucsPostProcessing
from flucs.utilities.messages import flucsprint

from flucs_fluid_itg.cold_itg_2d_fourier.profile_postprocessing import (
    load_profile_averages,
    parse_time_window,
    spectral_derivative,
)


def _warn_for_hyperdissipation(average, nc_path):
    for index in average.contributing_groups:
        hyper = average.input_files[index].get("hyperdissipation", {})
        if any(
            float(hyper.get(component, -1.0)) > 0.0
            for component in ("kx", "ky", "kperp", "kz")
        ):
            flucsprint(
                f"Generic hyperdissipation is active in {nc_path}; "
                "Pi_total omits the generic hyperdissipation contribution "
                "to the zonal-flow balance.",
                message_type="warning",
            )
            return


def _validate_zonal_flow_inputs(average):
    """Ensure the prescribed forcing is unchanged across restart groups."""
    reference = None
    for index in average.contributing_groups:
        forcing = average.input_files[index].get("forcing", {})
        values = (
            forcing.get("method", ""),
            forcing.get("momentum_flux_amplitude", 0.0),
            forcing.get("radial_mode_number", 1),
            forcing.get("radial_phase", 0.0),
            forcing.get("growth_rate", 1.0),
            forcing.get("midpoint_time", 0.0),
        )
        if reference is None:
            reference = values
            continue
        if values[0] != reference[0] or values[2] != reference[2]:
            raise ValueError(
                "Zonal-flow forcing parameters differ between groups."
            )
        if not np.allclose(values[1], reference[1]) or not np.allclose(
            values[3:], reference[3:]
        ):
            raise ValueError(
                "Zonal-flow forcing parameters differ between groups."
            )


def plot_momentum_flux(
    post,
    *,
    groups=None,
    time=None,
    fraction=None,
):
    """Create one decomposition-and-balance figure per simulation."""

    nc_paths = post.get_valid_netcdf_paths("momentum_flux/Pi_phi")
    for nc_path in nc_paths:
        average = load_profile_averages(
            post,
            nc_path,
            {
                "Pi_phi": "momentum_flux/Pi_phi",
                "Pi_T": "momentum_flux/Pi_T",
                "Pi_t": "momentum_flux/Pi_t",
                "Pi_d": "momentum_flux/Pi_d",
                "Pi_ZF": "momentum_flux/Pi_ZF",
                "Pi_total": "momentum_flux/Pi_total",
                "phi": "zonal_profiles/phi",
            },
            groups=groups,
            time_min=None if time is None else time[0],
            time_max=None if time is None else time[1],
            fraction=fraction,
        )
        _validate_zonal_flow_inputs(average)
        _warn_for_hyperdissipation(average, nc_path)

        x = average.x
        pi_phi = average.values["Pi_phi"]
        pi_temperature = average.values["Pi_T"]
        pi_turbulent = average.values["Pi_t"]
        pi_dissipative = average.values["Pi_d"]
        pi_zonal_flow = average.values["Pi_ZF"]
        pi_total = average.values["Pi_total"]
        shear = spectral_derivative(average.values["phi"], x, order=2)

        fig, axes = plt.subplots(
            2, 1, sharex=True, layout="constrained", figsize=(7, 7)
        )
        run_name = pl.Path(nc_path).parent.name
        figure_name = f"momentum_flux_{run_name}"
        fig.canvas.manager.set_window_title(figure_name)
        fig.suptitle(
            rf"{run_name}: $t\in[{average.time_min:.4g},"
            rf"{average.time_max:.4g}]$"
        )

        axes[0].plot(x, pi_phi, label=r"$\Pi_\phi$")
        axes[0].plot(x, pi_temperature, label=r"$\Pi_T$")
        axes[0].plot(x, pi_turbulent, label=r"$\Pi_t$")
        axes[0].plot(x, pi_zonal_flow, label=r"$\Pi_{ZF}$")
        axes[0].plot(x, shear, linestyle="--", label=r"$S$")
        axes[0].set_ylabel("decomposition")
        axes[0].legend(ncols=2)

        axes[1].plot(x, pi_turbulent, label=r"$\Pi_t$")
        axes[1].plot(x, pi_dissipative, label=r"$\Pi_d$")
        axes[1].plot(x, pi_zonal_flow, label=r"$\Pi_{ZF}$")
        axes[1].plot(
            x,
            pi_total,
            color="black",
            linestyle="--",
            label=r"$\Pi_{total}$",
        )
        axes[1].axhline(0.0, color="0.5", linestyle="dotted")
        axes[1].set_ylabel("momentum flux")
        axes[1].set_xlabel(r"$x$")
        axes[1].legend()

        for axis in axes:
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
        description="Plot time-averaged zonal momentum-flux profiles.",
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
    plot_momentum_flux(
        post,
        groups=args.groups,
        time=args.time,
        fraction=args.fraction,
    )
