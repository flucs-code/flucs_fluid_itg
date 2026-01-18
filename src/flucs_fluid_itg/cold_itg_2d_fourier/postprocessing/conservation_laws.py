import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from flucs.postprocessing import FlucsPostProcessing


def free_energy_check(post):
    # Get valid files for the specified variable
    nc_paths = post.get_valid_files("dWdt")

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Separate figure for each output
        fig, (ax, ax_error) = plt.subplots(2, 1, layout='constrained')

        figure_name = f"dWdt_check: {pl.Path(nc_path)}"
        fig.canvas.manager.set_window_title(figure_name)

        # Read data from netCDF file
        time, boundaries = post.load_netcdf_variable(nc_path, "time")
        dt, _ = post.load_netcdf_variable(nc_path, "dt")
        dWdt, _ = post.load_netcdf_variable(nc_path, "dWdt")
        injection, _ = post.load_netcdf_variable(nc_path, "dWdt_inj")
        dissipation, _ = post.load_netcdf_variable(nc_path, "dWdt_coll")

        for index in boundaries:
            ax.axvline(time[index], color='k', linestyle="dotted")
            ax_error.axvline(time[index], color='k', linestyle="dotted")

        # Plot data
        ax.plot(time, dWdt,
                label="dW/dt", linewidth=1.5, color='k', linestyle='solid')
        ax.plot(time, injection, label="injection", linewidth=1.5, color='r', linestyle='solid')
        ax.plot(time, dissipation, label="dissipation", linewidth=1.5, color='b', linestyle='solid')
        ax.plot(time, injection + dissipation, label="sum of inj and diss", linewidth=1.5, color='k', linestyle='dashed')

        error = (dWdt - injection - dissipation)/dt
        ax_error.plot(time[1:], error[1:], label="error / dt", linewidth=1.5, color='k')

        # Setting plot options
        ax.set_xlabel(r"$(2c_s/L_B)t$")
        ax_error.set_xlabel(r"$(2c_s/L_B)t$")

        ax.set_xlim(np.min(time), np.max(time))
        ax_error.set_xlim(np.min(time), np.max(time))
        # ax.set_ylim(ymin=0.0)

        ax.legend()
        ax_error.legend()

        # Save figures if required
        post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi": 300, "close": True})

        plt.show()


    return

if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()], 
        description="Check the conservation laws of the 2D ITG system.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_file="output.0d.nc",
        constraint="both"
    )

    # Call function
    free_energy_check(post)
