from collections.abc import Callable

import cupy as cp
import numpy as np
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions
from flucs.utilities.cupy import KernelWrapper


def _profile_variable(name, x):
    return FlucsDiagnosticVariable(
        name=name,
        shape=("x",),
        dimensions={"x": x},
        is_complex=False,
    )


class ZonalProfilesDiag(FlucsDiagnostic):
    """Zonal potential and temperature profiles on the padded x grid."""

    name = "zonal_profiles"

    def init_vars(self):
        x = (
            np.arange(self.system.nx, dtype=self.system.float)
            * self.system.float(self.system.input["dimensions.Lx"])
            / self.system.float(self.system.nx)
        )
        self.add_var(_profile_variable("phi", x))
        self.add_var(_profile_variable("T", x))
        self.add_var(
            FlucsDiagnosticVariable(
                name="gamma_max",
                shape=(),
                dimensions={},
                is_complex=False,
            )
        )

        self.zonal_fourier = cp.zeros(
            (2, self.system.nx), dtype=self.system.complex
        )
        self.zonal_fourier_host = np.empty(
            (2, self.system.nx), dtype=self.system.complex
        )

    def register_kernels(self):
        grid = (
            (self.system.nx + self.system.cuda_block_size - 1)
            // self.system.cuda_block_size,
        )
        self.gather_zonal_fields_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="gather_zonal_fields",
            grid=grid,
            block=(self.system.cuda_block_size,),
        )

    def ready(self):
        eigvals = self.system.compute_linear_eigensystem()["eigvals"]
        solved = self.system.get_solved_grid_mask().astype(bool)
        self.gamma_max = self.system.float(
            np.max(eigvals[:, solved].imag)
        )

    def execute(self):
        self.gather_zonal_fields_kernel(
            self.system.get_fields(), self.zonal_fourier
        )
        self.zonal_fourier.get(out=self.zonal_fourier_host)
        profiles = np.fft.ifft(
            self.zonal_fourier_host, axis=-1, norm="forward"
        ).real.astype(self.system.float, copy=False)

        self.save_data("phi", profiles[0])
        self.save_data("T", profiles[1])
        self.save_data("gamma_max", self.gamma_max)


class MomentumFluxDiag(FlucsDiagnostic):
    """Zonal turbulent and collisional momentum-flux profiles."""

    name = "momentum_flux"

    def init_vars(self):
        x = (
            np.arange(self.system.nx, dtype=self.system.float)
            * self.system.float(self.system.input["dimensions.Lx"])
            / self.system.float(self.system.nx)
        )
        for name in (
            "Pi_phi",
            "Pi_T",
            "Pi_t",
            "Pi_d",
            "Pi_AE",
            "Pi_total",
        ):
            self.add_var(_profile_variable(name, x))

        self.momentum_flux_fourier = cp.zeros(
            (6, self.system.nx), dtype=self.system.complex
        )
        self.momentum_flux_fourier_host = np.empty(
            (6, self.system.nx), dtype=self.system.complex
        )

    def register_kernels(self):
        self.find_derivatives_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="find_momentum_flux_derivatives",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        self.find_products_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="find_momentum_flux_products",
            grid=(self.system.full_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        zonal_grid = (
            (self.system.nx + self.system.cuda_block_size - 1)
            // self.system.cuda_block_size,
        )
        self.gather_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="gather_momentum_flux",
            grid=zonal_grid,
            block=(self.system.cuda_block_size,),
        )

        def create_first_intermediates(
            current_dt,
            current_time,
            current_step,
            fields,
            memory_dict,
        ):
            self.find_derivatives_kernel(
                fields, memory_dict["first_intermediates_fourier"]
            )

        def create_second_intermediates(
            current_dt,
            current_time,
            current_step,
            calculate_cfl,
            memory_dict,
        ):
            self.find_products_kernel(
                memory_dict["first_intermediates_real"],
                memory_dict["second_intermediates_real"],
            )

        self.operation, self.products_fourier = (
            self.system.create_dealiased_operation(
                n_in=3,
                n_out=2,
                create_first_intermediates=create_first_intermediates,
                create_second_intermediates=create_second_intermediates,
                allocate_additional_memory=None,
                combine_first_and_second_intermediates=True,
            )
        )

    def ready(self):
        pass

    def execute(self):
        fields = self.system.get_fields()
        self.operation(
            self.system.current_dt,
            self.system.current_time,
            self.system.current_step,
            fields,
            calculate_cfl=False,
        )
        self.gather_kernel(
            fields,
            self.products_fourier,
            self.momentum_flux_fourier,
            self.system.float(self.system.current_time),
        )
        self.momentum_flux_fourier.get(out=self.momentum_flux_fourier_host)
        profiles = np.fft.ifft(
            self.momentum_flux_fourier_host, axis=-1, norm="forward"
        ).real.astype(self.system.float, copy=False)

        for index, name in enumerate(
            ("Pi_phi", "Pi_T", "Pi_t", "Pi_d", "Pi_AE", "Pi_total")
        ):
            self.save_data(name, profiles[index])


class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"
    get_heatflux: Callable[..., cp.ndarray]

    def init_vars(self):
        reductions = FourierReductions(self.system)

        self.add_var(FlucsDiagnosticVariable(
            name="heatflux",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_heatflux = reductions.get_reduction(
            reduction_output="scalar",
            functor="Heatflux_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )

    def ready(self):
        pass

    def execute(self):
        fields = self.system.get_fields()

        self.vars["heatflux"].data_cache.append(
            self.get_heatflux(fields).get().item()
        )


class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"
    get_W: Callable[..., cp.ndarray]
    get_dWdt_coll: Callable[..., cp.ndarray]
    get_heatflux: Callable[..., cp.ndarray]
    get_dWdt_hyperdissipation_component: Callable[..., cp.ndarray]

    def init_vars(self):
        reductions = FourierReductions(self.system)

        # Total free energy W
        self.add_var(FlucsDiagnosticVariable(
            name="W",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        self.get_W = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergy_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )

        # Numerical time derivative of W
        self.add_var(FlucsDiagnosticVariable(
            name="dWdt",
            shape=(),
            dimensions={},
            is_complex=False
        ))

        # Collisional dissipation of W
        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_coll",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_dWdt_coll = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyColl_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )

        # Injection of W
        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_inj",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_heatflux = reductions.get_reduction(
            reduction_output="scalar",
            functor="Heatflux_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )

        for component in self.system.hyperdissipation_components:
            if component == "kz": # The model is 2D
                continue

            self.add_var(FlucsDiagnosticVariable(
                name=f"dWdt_hyperdissipation_{component}",
                shape=(),
                dimensions={},
                is_complex=False
                )
            )

        self.get_dWdt_hyperdissipation_component = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyHyperdissipationComponent_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )

        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_error",
            shape=(),
            dimensions={},
            is_complex=False
        ))

    def ready(self):
        pass

    def execute(self):
        current_dt = self.system.float(self.system.current_dt)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.get_fields()
        fields_previous = self.system.get_fields(1)

        # W
        W = self.get_W(fields).get().item()
        self.save_data("W", W)

        # dW/dt
        W_prev = self.get_W(fields_previous)
        dWdt = (W - W_prev.get().item()) / current_dt
        self.save_data("dWdt", dWdt)

        # dW/dt_coll
        dWdt_coll = self.get_dWdt_coll(fields).get().item()
        self.save_data("dWdt_coll", dWdt_coll)

        # dW/dt_inj
        heatflux = self.get_heatflux(fields).get().item().real
        dWdt_inj = self.system.input["parameters.kappaT"] * heatflux
        self.save_data("dWdt_inj", dWdt_inj)

        # Hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for index, component in enumerate(self.system.hyperdissipation_components):
            if component == "kz":
                continue

            result = self.get_dWdt_hyperdissipation_component(
                fields, adaptive_rate, index
            )

            dWdt_hyperdissipation_component = -result.get().item()
            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component

        self.save_data(
            "dWdt_error",
            dWdt - dWdt_inj - dWdt_coll - dWdt_hyperdissipation_total,
        )
