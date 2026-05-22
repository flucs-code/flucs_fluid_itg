import cupy as cp
from collections.abc import Callable
import numpy as np
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable

class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"
    get_heatflux: Callable[..., cp.ndarray]

    def init_vars(self):
        self.add_var(FlucsDiagnosticVariable(
            name="heatflux",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_heatflux = self.system.create_reduction(
            reduction_name="heatflux",
            shape=(1, self.system.nx),
            data_kernel_name="heatflux_kx",
            is_complex=True,
        )

    def ready(self):
        pass

    def execute(self):
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        self.vars["heatflux"].data_cache.append(
            -self.get_heatflux(phi, T).get().item().real
        )


class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"
    get_W: Callable[..., cp.ndarray]
    get_dW: Callable[..., cp.ndarray]
    get_dWdt_coll: Callable[..., cp.ndarray]
    get_heatflux: Callable[..., cp.ndarray]
    get_W_hyperdissipation_perp: Callable[..., cp.ndarray]
    get_W_hyperdissipation_kx: Callable[..., cp.ndarray]
    get_W_hyperdissipation_ky: Callable[..., cp.ndarray]

    def init_vars(self):
        # Total free energy W
        self.add_var(FlucsDiagnosticVariable(
            name="W",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_W = self.system.create_reduction(
            reduction_name="free_energy",
            shape=(1, self.system.nx),
            data_kernel_name="free_energy_kx",
            is_complex=False,
        )

        # Numerical time derivative of W
        self.add_var(FlucsDiagnosticVariable(
            name="dWdt",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_dW = self.system.create_reduction(
            reduction_name="dW",
            shape=(1, self.system.nx),
            data_kernel_name="dW_kx",
            is_complex=False,
        )

        # Collisional dissipation of W
        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_coll",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_dWdt_coll = self.system.create_reduction(
            reduction_name="dWdt_coll",
            shape=(1, self.system.nx),
            data_kernel_name="free_energy_collisional_loss_kx",
            is_complex=True,
        )

        # Injection of W
        self.add_var(FlucsDiagnosticVariable(
            name="dWdt_inj",
            shape=(),
            dimensions={},
            is_complex=False
        ))
        self.get_heatflux = self.system.create_reduction(
            reduction_name="heatflux",
            shape=(1, self.system.nx),
            data_kernel_name="heatflux_kx",
            is_complex=True,
        )

        for component in ["perp", "kx", "ky"]:
            self.add_var(FlucsDiagnosticVariable(
                name=f"dWdt_hyperdissipation_{component}",
                shape=(),
                dimensions={},
                is_complex=False
                )
            )
            get_hyperdiss = self.system.create_reduction(
                reduction_name=f"W_hyperdissipation_{component}",
                shape=(1, self.system.nx),
                data_kernel_name=f"W_hyperdissipation_{component}_kx",
                is_complex=False,
            )
            setattr(self, f"get_W_hyperdissipation_{component}", get_hyperdiss)

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
        phi = self.system.phi[self.system.current_step % 2]
        T = self.system.T[self.system.current_step % 2]

        # W
        self.save_data(
            "W",
            self.get_W(fields).get().item()
        )

        # numerical dW/dt
        result = self.get_dW(fields, fields_previous)
        dWdt = result.get().item() / current_dt
        self.save_data("dWdt", dWdt)

        # dW/dt_coll
        dWdt_coll = self.get_dWdt_coll(T).get().item().real
        self.save_data("dWdt_coll", dWdt_coll)

        # dW/dt_inj
        result = self.get_heatflux(phi, T).get().item().real
        dWdt_inj = -self.system.input["parameters.kappaT"] * result
        self.save_data("dWdt_inj", dWdt_inj)

        # Hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for component in ["perp", "kx", "ky"]:
            reduction = getattr(self, f"get_W_hyperdissipation_{component}")
            result = reduction(
                fields, adaptive_rate
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
