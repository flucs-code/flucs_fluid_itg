import cupy as cp
from collections.abc import Callable
import numpy as np
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions

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
        fields = self.system.fields[self.system.current_step % 2]

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

        fields = self.system.fields[self.system.current_step % 2]
        fields_previous = self.system.fields[self.system.current_step % 2 - 1]

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
