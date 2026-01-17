import cupy as cp
import numpy as np

from flucs.diagnostic import FlucsDiagnostic


class HeatfluxDiagOld(FlucsDiagnostic):
    name = "heatflux_old"
    shape = ()
    dimensions_dict = {}

    # d/dy operator in Fourier space
    dy: cp.ndarray

    def ready(self):
        # Copy ky wavenumbers to GPU memory for faster multiplication
        self.dy\
            = cp.array(1j * np.broadcast_to(self.system.ky,
                                            self.system.half_unpadded_tuple))

    def get_data(self):
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        return -2 * cp.sum(self.dy * phi * cp.conjugate(T)).item().real

    def print_diagnostic(self):
        print(f"Heat flux at time step {self.system.current_step} (time {self.system.current_time}) is {self.data_cache[-1]}")

class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"
    shape = ()
    dimensions_dict = {}

    # d/dy operator in Fourier space
    dy: cp.ndarray
    temp: cp.ndarray
    result: cp.ndarray
    find_heatflux_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def ready(self):
        # Allocate temporary memory
        self.temp = cp.zeros((self.system.nx,), dtype=self.system.complex)
        self.result = cp.zeros((1,), dtype=self.system.complex)

        # Get kernels
        self.find_heatflux_kx_kernel = self.system.cupy_module.get_function("find_heatflux_kx")
        self.last_axis_sum_nx_kernel = self.system.cupy_module.get_function("last_axis_sum_nx")


    def get_data(self):
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        self.find_heatflux_kx_kernel(
                (self.system.nx,),
                (256,),
                (phi, T, self.temp),
                shared_mem=32 * self.system.complex().nbytes)

        self.last_axis_sum_nx_kernel(
                (1,),
                (256,),
                (self.temp, self.result),
                shared_mem=32 * self.system.complex().nbytes)

        return -self.result.item().real;

    def print_diagnostic(self):
        print(f"Heat flux at time step {self.system.current_step} (time {self.system.current_time}) is {self.data_cache[-1]}")

class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"
    shape = ()
    dimensions_dict = {}

    # d/dy operator in Fourier space
    dy: cp.ndarray
    temp: cp.ndarray
    result: cp.ndarray
    find_free_energy_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def ready(self):
        # Allocate temporary memory
        self.temp = cp.zeros((self.system.nx,), dtype=self.system.float)
        self.result = cp.zeros((1,), dtype=self.system.float)

        # Get kernels
        self.find_free_energy_kx_kernel = self.system.cupy_module.get_function("find_free_energy_kx")
        self.real_last_axis_sum_nx_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nx")


    def get_data(self):
        T = self.system.T[self.system.current_step % 2]

        # return cp.sum(cp.abs(T)**2).item()

        self.find_free_energy_kx_kernel(
                (self.system.nx,),
                (256,),
                (T, self.temp),
                shared_mem=32 * self.system.float().nbytes)

        return cp.sum(self.temp).item()

        self.real_last_axis_sum_nx_kernel(
                (1,),
                (256,),
                (self.temp, self.result),
                shared_mem=32 * self.system.float().nbytes)

        return self.result.item();

    def print_diagnostic(self):
        print(f"Heat flux at time step {self.system.current_step} (time {self.system.current_time}) is {self.data_cache[-1]}")

class FreeEnergyTimeDerivativeDiag(FlucsDiagnostic):
    name = "dW_dt"
    shape = ()
    dimensions_dict = {}

    # d/dy operator in Fourier space
    dy: cp.ndarray
    temp: cp.ndarray
    result: cp.ndarray
    find_dW_dt_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def ready(self):
        # Allocate temporary memory
        self.temp = cp.zeros((self.system.nx,), dtype=self.system.float)
        self.result = cp.zeros((1,), dtype=self.system.float)

        # Get kernels
        self.find_dW_dt_kx_kernel = self.system.cupy_module.get_function("find_dW_dt_kx")
        self.real_last_axis_sum_nx_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nx")


    def get_data(self):
        T_now = self.system.T[self.system.current_step % 2]
        T_prev = self.system.T[(self.system.current_step - 1) % 2]

        self.find_dW_dt_kx_kernel(
                (self.system.nx,),
                (256,),
                (T_now, T_prev, self.temp),
                shared_mem=32 * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (1,),
                (256,),
                (self.temp, self.result),
                shared_mem=32 * self.system.float().nbytes)

        return self.result.item();

    def print_diagnostic(self):
        print(f"Heat flux at time step {self.system.current_step} (time {self.system.current_time}) is {self.data_cache[-1]}")
