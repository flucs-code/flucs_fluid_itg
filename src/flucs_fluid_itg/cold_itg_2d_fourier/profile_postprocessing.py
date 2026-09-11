"""Shared CPU helpers for zonal-profile postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeAverage:
    """Result of restart-aware physical-time averaging."""

    values: dict[str, np.ndarray]
    duration: float
    time_min: float
    time_max: float
    contributing_groups: tuple[int, ...]


@dataclass(frozen=True)
class ProfileAverage:
    """Loaded profile average and its validated group metadata."""

    values: dict[str, np.ndarray]
    x: np.ndarray
    parameters: dict[str, float]
    gamma_max: float
    duration: float
    time_min: float
    time_max: float
    input_files: list[dict]
    contributing_groups: tuple[int, ...]


def parse_time_window(value: str) -> tuple[float, float]:
    """Parse a ``START:END`` time-window command-line value."""

    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError("time must have the form START:END")
    try:
        start, end = (float(piece) for piece in pieces)
    except ValueError:
        raise ValueError("START and END must both be floats") from None
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("START and END must both be finite")
    if end <= start:
        raise ValueError("END must be greater than START")
    return start, end


def _time_close(first: float, second: float) -> bool:
    scale = max(1.0, abs(first), abs(second))
    return bool(np.isclose(first, second, rtol=1e-12, atol=1e-14 * scale))


def _validate_consistent(
    values: list,
    contributing: list[int],
    name: str,
) -> None:
    reference = np.asarray(values[contributing[0]])
    if not np.all(np.isfinite(reference)):
        raise ValueError(
            f"Non-finite {name!r} in a contributing output group."
        )
    for index in contributing[1:]:
        candidate = np.asarray(values[index])
        if (
            reference.shape != candidate.shape
            or not np.all(np.isfinite(candidate))
            or not np.allclose(
                reference, candidate, rtol=1e-10, atol=1e-12
            )
        ):
            raise ValueError(
                f"Inconsistent {name!r} across contributing output groups."
            )


def time_average_segments(
    times: list[np.ndarray],
    values: dict[str, list[np.ndarray]],
    *,
    time_min: float | None = None,
    time_max: float | None = None,
    fraction: float | None = None,
    grids: list[np.ndarray] | None = None,
    parameters: dict[str, list[float]] | None = None,
) -> TimeAverage:
    """Average group segments without integrating across restart boundaries."""

    if fraction is not None and (time_min is not None or time_max is not None):
        raise ValueError("--fraction cannot be combined with explicit bounds.")
    if fraction is not None and not 0.0 < fraction <= 1.0:
        raise ValueError(
            "--fraction must be greater than zero and at most one."
        )
    if not times:
        raise ValueError("No output groups were selected.")

    group_count = len(times)
    for name, group_values in values.items():
        if len(group_values) != group_count:
            raise ValueError(f"Variable {name!r} has the wrong group count.")
    if grids is not None and len(grids) != group_count:
        raise ValueError("The x grids have the wrong group count.")
    for name, group_parameters in (parameters or {}).items():
        if len(group_parameters) != group_count:
            raise ValueError(f"Parameter {name!r} has the wrong group count.")

    local_times = []
    local_values = {name: [] for name in values}
    previous_end = None
    for group_index, group_time in enumerate(times):
        time = np.asarray(group_time, dtype=float)
        if time.ndim != 1:
            raise ValueError("Time coordinates must be one-dimensional.")
        if time.size and (not np.all(np.isfinite(time))):
            raise ValueError("Time coordinates must be finite.")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("Times must increase strictly within each group.")

        if (
            time.size
            and previous_end is not None
            and time[0] < previous_end
            and not _time_close(float(time[0]), previous_end)
        ):
            raise ValueError(
                "Output groups overlap for positive duration; select "
                "non-overlapping groups explicitly with --groups."
            )

        # A shared endpoint has no duration of its own. Keeping it in each
        # adjacent segment preserves both one-sided trapezoids while still
        # avoiding any cross-group interval or double-counted duration.
        group_slice = slice(None)
        time = time[group_slice]
        local_times.append(time)
        for name, group_values in values.items():
            array = np.asarray(group_values[group_index])
            if array.shape[:1] != np.asarray(group_time).shape:
                raise ValueError(
                    f"Variable {name!r} does not match its time coordinate."
                )
            local_values[name].append(array[group_slice])

        original_time = np.asarray(group_time, dtype=float)
        if original_time.size:
            previous_end = float(original_time[-1])

    nonempty = [time for time in local_times if time.size]
    if not nonempty:
        raise ValueError("Selected output groups contain no samples.")
    available_min = min(float(time[0]) for time in nonempty)
    available_max = max(float(time[-1]) for time in nonempty)

    if fraction is None and time_min is None and time_max is None:
        fraction = 0.2
    if fraction is not None:
        lower = available_max - fraction * (available_max - available_min)
        upper = available_max
    else:
        lower = available_min if time_min is None else float(time_min)
        upper = available_max if time_max is None else float(time_max)

    lower = max(lower, available_min)
    upper = min(upper, available_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(
            "The selected averaging window has no positive duration."
        )

    numerators = {name: None for name in values}
    duration = 0.0
    contributing = []
    for group_index, time in enumerate(local_times):
        selected = (time >= lower) & (time <= upper)
        selected_time = time[selected]
        if selected_time.size < 2:
            continue

        contributing.append(group_index)
        segment_duration = float(selected_time[-1] - selected_time[0])
        duration += segment_duration
        for name, group_values in local_values.items():
            selected_values = group_values[group_index][selected]
            if not np.all(np.isfinite(selected_values)):
                raise ValueError(
                    f"Variable {name!r} contains non-finite samples in the "
                    "selected window."
                )
            integral = np.trapezoid(
                selected_values, selected_time, axis=0
            )
            if numerators[name] is None:
                numerators[name] = integral
            else:
                if np.shape(numerators[name]) != np.shape(integral):
                    raise ValueError(
                        f"Variable {name!r} has incompatible group shapes."
                    )
                numerators[name] = numerators[name] + integral

    if duration <= 0.0 or not contributing:
        raise ValueError(
            "The chosen window must contain at least two samples and have "
            "positive duration."
        )

    if grids is not None:
        _validate_consistent(grids, contributing, "x grid")
    for name, group_parameters in (parameters or {}).items():
        _validate_consistent(group_parameters, contributing, name)

    averages = {
        name: np.asarray(numerator) / duration
        for name, numerator in numerators.items()
    }
    return TimeAverage(
        values=averages,
        duration=duration,
        time_min=lower,
        time_max=upper,
        contributing_groups=tuple(contributing),
    )


def spectral_derivative(
    profile: np.ndarray,
    x: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Return a spectral derivative of a real periodic one-dimensional field."""

    profile = np.asarray(profile)
    x = np.asarray(x, dtype=float)
    if profile.ndim != 1 or x.ndim != 1 or profile.shape != x.shape:
        raise ValueError(
            "Profile and x must be matching one-dimensional arrays."
        )
    if order not in (1, 2):
        raise ValueError("Only first and second derivatives are supported.")
    if x.size < 2:
        raise ValueError("At least two x samples are required.")

    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("The x grid must be uniformly spaced.")
    wavenumber = 2.0 * np.pi * np.fft.fftfreq(x.size, d=spacing[0])
    multiplier = 1j * wavenumber if order == 1 else -(wavenumber**2)
    coefficients = np.fft.fft(profile, norm="forward")
    return np.fft.ifft(
        multiplier * coefficients, norm="forward"
    ).real.astype(profile.dtype, copy=False)


def load_profile_averages(
    post,
    nc_path,
    variables: dict[str, str],
    *,
    groups=None,
    time_min: float | None = None,
    time_max: float | None = None,
    fraction: float | None = None,
) -> ProfileAverage:
    """Load, validate, and average profile variables from a NetCDF output."""

    times = post.load_netcdf_variable(
        nc_path, "time", groups=groups, concatenate=False
    )[0]
    input_files = post.load_netcdf_input_files(nc_path, groups=groups)

    group_values = {}
    variable_grids = {}
    for short_name, variable in variables.items():
        data, _, dimensions = post.load_netcdf_variable(
            nc_path, variable, groups=groups, concatenate=False
        )
        group_values[short_name] = data
        variable_grids[short_name] = [
            dims.get("x", np.array([np.nan])) for dims in dimensions
        ]

    gamma_groups = post.load_netcdf_variable(
        nc_path,
        "zonal_profiles/gamma_max",
        groups=groups,
        concatenate=False,
    )[0]
    gamma_values = []
    for gamma in gamma_groups:
        gamma = np.asarray(gamma)
        if gamma.size == 0:
            gamma_values.append(np.nan)
        elif not np.allclose(
            gamma, gamma.flat[0], rtol=1e-10, atol=1e-12
        ):
            raise ValueError("gamma_max varies within one output group.")
        else:
            gamma_values.append(float(gamma.flat[0]))

    parameter_keys = {
        "kappaT": "kappaT",
        "chi": "chi",
        "a": "coeffa",
        "b": "coeffb",
    }
    parameters = {
        output_name: [
            float(input_file["parameters"][input_name])
            for input_file in input_files
        ]
        for output_name, input_name in parameter_keys.items()
    }
    consistency_values = {**parameters, "gamma_max": gamma_values}

    first_name = next(iter(variables))
    result = time_average_segments(
        times,
        group_values,
        time_min=time_min,
        time_max=time_max,
        fraction=fraction,
        grids=variable_grids[first_name],
        parameters=consistency_values,
    )

    reference_grid = variable_grids[first_name][
        result.contributing_groups[0]
    ]
    for short_name, grids in variable_grids.items():
        _validate_consistent(
            grids, list(result.contributing_groups), f"{short_name} x grid"
        )
        for index in result.contributing_groups:
            if not np.allclose(
                grids[index], reference_grid, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"Variable {short_name!r} uses an incompatible x grid."
                )

    first_group = result.contributing_groups[0]
    return ProfileAverage(
        values=result.values,
        x=np.asarray(reference_grid),
        parameters={
            name: values[first_group] for name, values in parameters.items()
        },
        gamma_max=gamma_values[first_group],
        duration=result.duration,
        time_min=result.time_min,
        time_max=result.time_max,
        input_files=input_files,
        contributing_groups=result.contributing_groups,
    )
