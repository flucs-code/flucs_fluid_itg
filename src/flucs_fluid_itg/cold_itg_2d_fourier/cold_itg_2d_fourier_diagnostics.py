import cupy as cp
import numpy as np
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
#TODO remove these when coding optimisation wrapper
BLOCK_SIZE = int(256)
THREADS_PER_WARP = int(32) 

class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"

    temp: cp.ndarray
    result: cp.ndarray
    heatflux_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def init_vars(self):
        self.add_var(FlucsDiagnosticVariable(
            name="heatflux",
            shape=(),
            dimensions={},
            is_complex=False
        ))

    def ready(self):
        # Allocate temporary memory
        self.temp = cp.zeros((self.system.nx,), dtype=self.system.complex)
        self.result = cp.zeros((1,), dtype=self.system.complex)

        # Get kernels
        self.heatflux_kx_kernel = self.system.cupy_module.get_function("heatflux_kx")
        self.last_axis_sum_nx_kernel = self.system.cupy_module.get_function("last_axis_sum_nx")


    def execute(self):
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        self.heatflux_kx_kernel(
                (self.system.nx,),
                (BLOCK_SIZE,),
                (phi, T, self.temp),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nx_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.temp, self.result),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.vars["heatflux"].data_cache.append(-self.result.item().real)


class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"

    temp: cp.ndarray
    result: cp.ndarray
    free_energy_kx_kernel: cp.RawKernel
    last_axis_sum_nx_kernel: cp.RawKernel

    def init_vars(self):
        self.add_var(FlucsDiagnosticVariable(
            name="W",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_coll",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_inj",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_perp",
            shape=(), 
            dimensions={}, 
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_kx",
            shape=(), 
            dimensions={}, 
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_ky",
            shape=(), 
            dimensions={}, 
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_hyperdissipation_kz",
            shape=(), 
            dimensions={}, 
            is_complex=False
            )
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_error",
            shape=(),
            dimensions={},
            is_complex=False
        ))

    def ready(self):
        # Allocate temporary memory
        self.real_temp = cp.zeros((self.system.nx,), dtype=self.system.float)
        self.real_result = cp.zeros((1,), dtype=self.system.float)

        self.complex_temp = cp.zeros((self.system.nx,), dtype=self.system.complex)
        self.complex_result = cp.zeros((1,), dtype=self.system.complex)

        # Get kernels
        self.heatflux_kx_kernel = self.system.cupy_module.get_function("heatflux_kx")
        self.dW_kx_kernel = self.system.cupy_module.get_function("dW_kx")
        self.free_energy_kx_kernel = self.system.cupy_module.get_function("free_energy_kx")
        self.free_energy_collisional_loss_kx_kernel = self.system.cupy_module.get_function("free_energy_collisional_loss_kx")
        self.last_axis_sum_nx_kernel = self.system.cupy_module.get_function("last_axis_sum_nx")
        self.real_last_axis_sum_nx_kernel = self.system.cupy_module.get_function("real_last_axis_sum_nx")

        self.hyperdissipation_magnitude_kernels = {
            "perp": self.system.cupy_module.get_function("W_hyperdissipation_perp_kx"),
            "kx": self.system.cupy_module.get_function("W_hyperdissipation_kx_kx"),
            "ky": self.system.cupy_module.get_function("W_hyperdissipation_ky_kx"),
            "kz": self.system.cupy_module.get_function("W_hyperdissipation_kz_kx"),
        }

    def execute(self):
        fields = self.system.fields[self.system.current_step % 2]
        fields_previous = self.system.fields[self.system.current_step % 2 - 1]
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        # W
        self.free_energy_kx_kernel(
                (self.system.nx,),
                (BLOCK_SIZE,),
                (fields, self.real_temp),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.real_temp, self.real_result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.save_data("W", self.real_result.get().item())

        # dW/dt
        self.dW_kx_kernel(
                (self.system.nx,),
                (BLOCK_SIZE,),
                (fields, fields_previous, self.real_temp),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        self.real_last_axis_sum_nx_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.real_temp, self.real_result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

        dWdt = self.real_result.get().item() / self.system.current_dt
        self.save_data("dWdt", dWdt)

        # dW/dt_coll
        self.free_energy_collisional_loss_kx_kernel(
                (self.system.nx,),
                (BLOCK_SIZE,),
                (T, self.complex_temp),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nx_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.complex_temp, self.complex_result),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        dWdt_coll = self.complex_result.real.get().item()
        self.save_data("dWdt_coll", dWdt_coll)

        # dW/dt_inj

        self.heatflux_kx_kernel(
                (self.system.nx,),
                (BLOCK_SIZE,),
                (phi, T, self.complex_temp),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        self.last_axis_sum_nx_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.complex_temp, self.complex_result),
                shared_mem=THREADS_PER_WARP * self.system.complex().nbytes)

        dWdt_inj = -self.system.input["parameters.kappaT"] * self.complex_result.get().item().real
        self.save_data("dWdt_inj", dWdt_inj)

        # Hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for component, kernel in self.hyperdissipation_magnitude_kernels.items():

            kernel(
                (self.system.nx,),
                (BLOCK_SIZE,),
                (fields, self.real_temp),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

            self.real_last_axis_sum_nx_kernel(
                (1,),
                (BLOCK_SIZE,),
                (self.real_temp, self.real_result),
                shared_mem=THREADS_PER_WARP * self.system.float().nbytes)

            dWdt_hyperdissipation_component = -self.real_result.get().item()
            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component

        self.save_data(
            "dWdt_error",
            dWdt - dWdt_inj - dWdt_coll - dWdt_hyperdissipation_total,
        )
