"""Forcing methods specific to the cold-ion ITG system."""

import numpy as np
from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing


class ColdITGAlfvenEigenmodeForcing(FourierSystemForcing):
    """Prescribed zonal momentum flux from a growing Alfvén mode pair.

    This is a one-way surrogate: it supplies the Reynolds-stress-like source
    that an Alfvén eigenmode self-interaction would exert on the zonal flow,
    without evolving an additional Alfvén envelope.
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
        if mode < 1 or mode >= system.half_nx_unpadded:
            raise InvalidFlucsInputFileError(
                "forcing.radial_mode_number must be a retained nonzero radial "
                "mode."
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

        system.module_options.define_int("FORCING_AE_MODE", mode)
        system.module_options.define_float("FORCING_AE_AMPLITUDE", amplitude)
        system.module_options.define_float("FORCING_AE_PHASE", phase)
        system.module_options.define_float(
            "FORCING_AE_GROWTH_RATE", growth_rate
        )
        system.module_options.define_float(
            "FORCING_AE_MIDPOINT_TIME", midpoint
        )
