"""Forcing methods specific to the cold-ion ITG system."""

import numpy as np
from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing


class ZonalFlowForcing(FourierSystemForcing):
    """
    Drive the zonal potential with a prescribed momentum flux.

    The forcing is an explicit source for phi at ky = 0. It prescribes
    one real radial Fourier mode of the zonal momentum flux and leaves the
    temperature and all nonzonal modes unforced. Its time-dependent amplitude
    is multiplied by the logistic envelope

    1 / (1 + exp(-2 * growth_rate * (time - midpoint_time))).

    Set forcing.method = "zonal_flow" to enable it. The forcing parameters
    are:

    * momentum_flux_amplitude: nonnegative magnitude of the fully grown
      real-space momentum-flux profile.
    * radial_mode_number: positive, retained radial Fourier-mode number.
    * radial_phase: phase of the radial profile, in radians.
    * growth_rate: positive rate controlling the logistic turn-on.
    * midpoint_time: time at which the envelope reaches one half.

    With amplitude A, mode m, and phase p, the prescribed profile
    approaches -A * sin(2*pi*m*x/Lx + p) after the turn-on. The same profile
    is exposed by the momentum-flux diagnostic as Pi_ZF_forcing and is
    included in Pi_total.
    """

    linear = False
    explicit = True

    def setup_cuda_definitions(self) -> None:
        system = self.system
        input_data = system.input

        amplitude = input_data["forcing.momentum_flux_amplitude"]
        mode = input_data["forcing.radial_mode_number"]
        phase = input_data["forcing.radial_phase"]
        growth_rate = input_data["forcing.growth_rate"]
        midpoint = input_data["forcing.midpoint_time"]

        if not np.isfinite(amplitude) or amplitude < 0.0:
            raise InvalidFlucsInputFileError(
                "forcing.momentum_flux_amplitude must be finite and nonnegative."
            )
        if (
            not isinstance(mode, int)
            or isinstance(mode, bool)
            or mode < 1
            or mode >= system.half_nx_unpadded
        ):
            raise InvalidFlucsInputFileError(
                "forcing.radial_mode_number must be an integer corresponding "
                "to a retained nonzero radial mode."
            )
        if not np.isfinite(phase):
            raise InvalidFlucsInputFileError(
                "forcing.radial_phase must be finite."
            )
        if not np.isfinite(growth_rate) or growth_rate <= 0.0:
            raise InvalidFlucsInputFileError(
                "forcing.growth_rate must be finite and positive."
            )
        if not np.isfinite(midpoint):
            raise InvalidFlucsInputFileError(
                "forcing.midpoint_time must be finite."
            )

        system.module_options.define_int("FORCING_ZONAL_FLOW_MODE", mode)
        system.module_options.define_float(
            "FORCING_ZONAL_FLOW_AMPLITUDE", amplitude
        )
        system.module_options.define_float("FORCING_ZONAL_FLOW_PHASE", phase)
        system.module_options.define_float(
            "FORCING_ZONAL_FLOW_GROWTH_RATE", growth_rate
        )
        system.module_options.define_float(
            "FORCING_ZONAL_FLOW_MIDPOINT_TIME", midpoint
        )
