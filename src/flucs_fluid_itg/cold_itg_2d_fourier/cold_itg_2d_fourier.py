"""
Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.
"""
from typing import ClassVar

import cupy as cp
import numpy as np
from cupy.cuda import cufft
from flucs.input import InvalidFlucsInputFileError
from flucs.diagnostic import FlucsDiagnostic
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper

from .cold_itg_2d_fourier_diagnostics import FreeEnergyDiag, HeatfluxDiag


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    number_of_fields = 2
    number_of_fields_explicit = 2
    number_of_dft_derivatives = 5
    number_of_dft_bits = 5

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    real_dxphi: cp.ndarray
    real_dxphi_zonal: cp.ndarray

    # CUDA kernels
    find_derivatives: KernelWrapper
    find_nonlinear_bits: KernelWrapper
    zonal_average: KernelWrapper

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
        zonal_average_cuda_grid = (self.padded_nx,)
        zonal_average_shared_mem = 32 * self.float().nbytes

        nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

        # System-specific kernels
        self.find_derivatives = KernelWrapper(
            system=self,
            cuda_kernel_name="find_derivatives",
            grid=(self.half_padded_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        self.find_nonlinear_bits = KernelWrapper(
            system=self,
            cuda_kernel_name="find_nonlinear_bits",
            grid=(self.full_padded_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=nonlinear_bits_shared_mem,
        )

        self.zonal_average = KernelWrapper(
            system=self,
            cuda_kernel_name="last_axis_average_float",
            grid=zonal_average_cuda_grid,
            block=zonal_average_cuda_block,
            shared_mem=zonal_average_shared_mem,
        )

    def _allocate_memory(self) -> None:
        """Allocates runtime arrays."""

        # First, call FourierSystem's method which allocates
        # self.fields among other things.
        super()._allocate_memory(allocate_derivatives_and_bits=True,
                                 combine_derivatives_and_bits=True)

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

            # The first derivative in real_derivatives is dx phi.
            # We need this for computing the zonal flow.
            self.real_dxphi = cp.ndarray((self.padded_nx, self.padded_ny),
                                         dtype=self.float,
                                         memptr=self.real_derivatives.data)

            self.real_dxphi_zonal = cp.zeros((self.padded_nx,),
                                             dtype=self.float)

    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Anything custom goes here

        if self.nz != 1 or self.padded_nz != 1:
            raise ValueError("Both nz and padded_nz should be "
                             "set to 1 for the 2D system!")

        if self.input["hyperdissipation.kz"] > 0.0:
            raise InvalidFlucsInputFileError(
                "Hyperdissipation in kz is not supported for 2D systems."
            )

    def setup_cuda_defs(self) -> None:
        # System-specific constants for the kernels
        self.module_options.define_float("CHI",
                                            self.input["parameters.chi"])
        self.module_options.define_float("A_TIMES_CHI",
                                            self.input["parameters.a"]
                                            * self.input["parameters.chi"])

        self.module_options.define_float("B_TIMES_CHI",
                                            self.input["parameters.b"]
                                            * self.input["parameters.chi"])

        self.module_options.define_float("KAPPA_T",
                                            self.input["parameters.kappaT"])
        self.module_options.define_float("KAPPA_N",
                                            self.input["parameters.kappan"])
        self.module_options.define_float("KAPPA_B",
                                            self.input["parameters.kappaB"])

        # Call this to compile the module
        super().setup_cuda_defs()

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def calculate_nonlinear_terms(self) -> None:
        """
        Calculates the nonlinear terms. This is the most computationaly
        intensive part of taking a time step. Here, we also determine the
        nonlinear CFL coefficient.

        """
        self.find_derivatives(
            self.fields[self.current_step % 2 - 1],
            self.dft_derivatives,
            self.real_dxphi_zonal,
            self.cfl_rate
        )

        self.plan_derivatives_c2r.fft(self.dft_derivatives,
                                      self.real_derivatives,
                                      cufft.CUFFT_INVERSE)

        self.zonal_average(
            self.padded_ny,
            False,
            self.real_dxphi,
            self.real_dxphi_zonal
        )

        # NB: real_derivatives and real_bits are the same array
        self.find_nonlinear_bits(
            self.real_derivatives,
            self.real_dxphi_zonal,
            self.cfl_rate,
        )

        # NB: real_derivatives and real_bits are the same array
        self.plan_bits_r2c.fft(self.real_bits, self.dft_bits, cufft.CUFFT_FORWARD)

        super().calculate_nonlinear_terms()

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_linear_matrix_reference(self) -> np.ndarray:

        # Initialise linear matrix
        linear_matrix = np.zeros(
            (
            self.number_of_fields, 
            self.number_of_fields, 
            *self.half_unpadded_tuple
            ),
            dtype=self.complex,
        )

        # Get wavenumbers
        kx, ky, kz = self.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2

        # Get parameters
        kappaT = self.input["parameters.kappaT"]
        kappaB = self.input["parameters.kappaB"]
        kappan = self.input["parameters.kappan"]

        chi = self.input["parameters.chi"]
        a = self.input["parameters.a"]
        b = self.input["parameters.b"]

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
