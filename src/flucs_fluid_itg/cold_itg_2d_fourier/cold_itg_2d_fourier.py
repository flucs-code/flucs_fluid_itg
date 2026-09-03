"""
Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.
"""
from typing import ClassVar

import cupy as cp
import numpy as np
from flucs.input import InvalidFlucsInputFileError
from flucs.diagnostic import FlucsDiagnostic
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper

from .cold_itg_2d_fourier_diagnostics import FreeEnergyDiag, HeatfluxDiag


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    number_of_fields = 2
    number_of_dft_derivatives = 5
    number_of_dft_bits = 5

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    real_dxphi_zonal: cp.ndarray

    # CUDA kernels
    find_derivatives_kernel: KernelWrapper
    find_nonlinear_bits_kernel: KernelWrapper
    zonal_average_kernel: KernelWrapper

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        HeatfluxDiag, FreeEnergyDiag
    }

    def ready(self):
        # Anything system-specific goes here
        super().ready()

    def register_kernels(self) -> None:
        super().register_kernels()

        # Setup kernel parameters (grid, block, shared memory)
        zonal_average_cuda_block = (256,)
        zonal_average_cuda_grid = (self.nx,)
        zonal_average_shared_mem = 32 * self.float().nbytes

        nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

        # System-specific kernels
        self.find_derivatives_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_derivatives",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        self.find_nonlinear_bits_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_nonlinear_bits",
            grid=(self.full_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=nonlinear_bits_shared_mem,
        )

        self.zonal_average_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="last_axis_average_float",
            grid=zonal_average_cuda_grid,
            block=zonal_average_cuda_block,
            shared_mem=zonal_average_shared_mem,
        )

        # Define functions from kernels
        def find_derivatives_function(
            current_dt,
            current_time,
            current_step: int,
            fields: cp.ndarray,
            memory_dict: dict
        ) -> None:
            fourier_derivatives = memory_dict["first_intermediates_fourier"]
            self.find_derivatives_kernel(
                fields,
                fourier_derivatives,
            )

        def find_nonlinear_bits_function(
            current_dt,
            current_time,
            current_step: int,
            calculate_cfl: bool,
            memory_dict: dict
        ) -> None:
            # Get dxphi in whichever array (shifted or unshifted) is
            # currently being evaluated.
            real_derivatives = memory_dict["first_intermediates_real"]
            real_bits = memory_dict["second_intermediates_real"]

            real_dxphi = real_derivatives[0]

            self.zonal_average_kernel(
                self.ny,
                False,
                real_dxphi,
                self.real_dxphi_zonal,
            )

            self.find_nonlinear_bits_kernel(
                real_derivatives,
                real_bits,
                self.real_dxphi_zonal,
                calculate_cfl,
                self.cfl_rate,
            )

        if not self.input["setup.linear"]:
            self.dft_derivatives_operation, self.dft_bits = (
                self.create_dealiased_operation(
                    n_in=self.number_of_dft_derivatives,
                    n_out=self.number_of_dft_bits,
                    create_first_intermediates=find_derivatives_function,
                    create_second_intermediates=find_nonlinear_bits_function,
                    allocate_additional_memory=None,
                    combine_first_and_second_intermediates=True,
                )
            )

    def _allocate_memory(self) -> None:
        """Allocates runtime arrays."""

        # First, call FourierSystem's method which allocates
        # self.fields among other things.
        super()._allocate_memory()

        # GPU arrays

        # Pointers to phi and T for easier access
        self.phi = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][0, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][0, 0, 0, 0].data),]

        self.T = [cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[0][1, 0, 0, 0].data),
                  cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[1][1, 0, 0, 0].data),]

        if not self.input["setup.linear"]:
            # All fields and derivatives to be transformed to real space
            # are kept in one huge array (dft_derivatives).
            # The first index indexes the fields and it's meaning is
            # 0 dxphi,
            # 1 dyphi,
            # 2 (dx^2 - dy^2) phi,
            # 3 dxdyphi
            # 4 p

            # The NL bits here are
            # 0 dxphi0 * dyphi
            # 1 (dx^2 - dy^2)phi * p
            # 2 dxdyphi * p
            # 3 dxphi * p
            # 4 dyphi * p

            # The arrays for the above are handled by FourierSystem
            # Here follow all the ITG-specific ones

            self.real_dxphi_zonal = cp.zeros((self.nx,),
                                             dtype=self.float)

    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        if self.nz != 1:
            raise ValueError("nz must be set to 1 for the 2D system!")

        if self.input["hyperdissipation.kz"] > 0.0:
            raise InvalidFlucsInputFileError(
                "Hyperdissipation in kz is not supported for 2D systems."
            )

    def setup_cuda_definitions(self) -> None:
        # System-specific constants for the kernels
        self.module_options.define_float("CHI",
                                         self.input["parameters.chi"])
        self.module_options.define_float("COEFFA_TIMES_CHI",
                                         self.input["parameters.coeffa"]
                                         * self.input["parameters.chi"])

        self.module_options.define_float("COEFFB_TIMES_CHI",
                                         self.input["parameters.coeffb"]
                                         * self.input["parameters.chi"])

        self.module_options.define_float("KAPPA_T",
                                         self.input["parameters.kappaT"])
        self.module_options.define_float("KAPPA_N",
                                         self.input["parameters.kappan"])
        self.module_options.define_float("KAPPA_B",
                                         self.input["parameters.kappaB"])

        # Call this to compile the module
        super().setup_cuda_definitions()

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def compute_nonlinear_terms(
        self,
        current_dt,
        current_time,
        current_step,
        fields: cp.ndarray,
        calculate_cfl,
    ) -> None:
        """Computes the dealiased nonlinear terms for the supplied fields."""
        self.dft_derivatives_operation(
            current_dt,
            current_time,
            current_step,
            fields,
            calculate_cfl=calculate_cfl,
        )

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_linear_matrix_reference(self) -> np.ndarray:

        # Initialise linear matrix
        linear_matrix = np.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                *self.half_tuple
            ),
            dtype=self.complex,
        )

        # Get wavenumbers
        _, kx, ky = self.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2

        # Get parameters
        kappaT = self.input["parameters.kappaT"]
        kappaB = self.input["parameters.kappaB"]
        kappan = self.input["parameters.kappan"]

        chi = self.input["parameters.chi"]
        a = self.input["parameters.coeffa"]
        b = self.input["parameters.coeffb"]

        # Define arrays for zonal response
        eta = 1 + kperp2
        eta[0, :, 0] = kperp2[0, :, 0]
        eta[0, 0, 0] = 1.0

        # phi-phi
        linear_matrix[0, 0, :, :, :] = (
                    + a * chi * (kperp2**2)
                    - 1j * (kappaB - kappan) * ky
                    - 1j * kappaT * kperp2 * ky
                    ) / eta

        # phi-T
        linear_matrix[0, 1, :, :, :] = (
                    - b * chi * (kperp2**2)
                    - 1j * kappaB * ky) / eta

        # T-phi
        linear_matrix[1, 0, :, :, :] = 1j * kappaT * ky

        # T-T
        linear_matrix[1, 1, :, :, :] = chi*kperp2

        return linear_matrix
