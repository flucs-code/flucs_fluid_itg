# AGENTS.md

## Scope

This repository is a small `flucs` system plugin. It implements the cold-ion,
two-dimensional fluid ITG model based on equations (2.17) and (2.18) of Ivanov
et al. (2020), *The Dimits regime of curvature-driven ion-temperature-gradient
turbulence* (doi:10.1017/S0022377820000938). Most changes here should add or
improve diagnostics and postprocessing without disturbing the working solver.

The companion framework checkout is called `flucs` and its location is system-
dependent. Ask the user in case you cannot find it automatically (e.g., by
checking the installed python packages). Read the relevant implementation
there before changing an inherited interface. This plugin and that checkout
usually need matching development branches; the dependency in `pyproject.toml`
is deliberately unpinned, so compatibility is not enforced by packaging.

## Repository map and entry points

- `pyproject.toml` registers `ColdITG2DFourier` in the `flucs.systems` entry-point
  group. The core package registers `FourierSolver` and the `flucs` CLI.
- `src/flucs_fluid_itg/cold_itg_2d_fourier/cold_itg_2d_fourier.py` defines the
  Python system: input checks, allocation, dealiased-operation callbacks,
  kernel registration, compile-time parameters, and a NumPy reference linear
  matrix.
- The adjacent `.cu` file supplies the model-specific device functions and
  kernels. It includes `flucs/solvers/fourier/fourier_system.cuh`, which supplies
  indexing, reductions, hyperdissipation, Padé propagators, and timestepper
  integration points.
- The adjacent `.toml` adds model defaults on top of `FlucsSystem` and
  `FourierSystem` defaults. Defaults are loaded through the system MRO, based on
  module and TOML stem names; preserve this naming/layout convention.
- `cold_itg_2d_fourier_diagnostics.py` defines the system runtime diagnostics.
- `profile_postprocessing.py` contains CPU-only, restart-aware time-averaging
  and spectral-derivative helpers shared by the profile plotting scripts.
- `cold_itg_2d_fourier/postprocessing/*.py` contains discoverable standalone
  postprocessing scripts. `flucs -p` finds both solver and system scripts by
  inspecting the installed class locations.
- `_version.py`, egg-info, caches, and bytecode are generated artifacts. Do not
  hand-edit or commit them.

Install both repositories editable in the same environment during development.
Use `flucs --list` to confirm that `FourierSolver` and `ColdITG2DFourier` are
visible. A run directory contains `input.toml`; its `[setup]` must select those
entry-point names.

## Model and numerical architecture

The evolved fields are Fourier coefficients of electrostatic potential `phi`
(field 0) and ion-temperature perturbation `T` (field 1); `p = phi + T` is used
in the nonlinear products. The model is strictly 2-D: `nz` must equal 1 and
positive `hyperdissipation.kz` is rejected. The modified adiabatic-electron
response makes the generalized-vorticity factor `eta` equal to `kperp^2` for
zonal (`ky == 0`) modes and `1 + kperp^2` for nonzonal modes. The all-zero mode
is specially protected from division by zero.

The paper's two main physical parameters are `kappaT` and `chi`. This
implementation also exposes `kappaB`, `kappan`, and independently named
collisional coefficients `coeffa` and `coeffb`; the CUDA definitions use
`COEFFA_TIMES_CHI` and `COEFFB_TIMES_CHI`. The default `alpha` parameter is
currently unused. Do not remove or reinterpret it casually because existing
input files may contain it.

The linear operator is evaluated per Fourier mode in `get_linear_matrix` and
advanced by the core Fourier timesteppers using a Padé-approximated exponential
propagator. `compute_linear_matrix_reference()` must remain an independent
NumPy expression of the same matrix so `setup.check_linear_matrix` can detect
CUDA/reference drift. Keep changes to either implementation paired and check
the sign convention in the timestepper: the framework propagates the matrix
and subtracts explicit terms according to its equation convention.

Nonlinear terms are pseudospectral and explicitly integrated. The system asks
`FourierSystem.create_dealiased_operation()` for the configured two-thirds or
phase-shift path. Five Fourier intermediates are transformed to real space:

1. `dx(phi)`
2. `dy(phi)`
3. `(dx^2 - dy^2) phi`
4. `dx dy phi`
5. `p = phi + T`

The real-space kernel forms five products. One uses the y-average of `dx(phi)`
for the zonal-flow contribution; the other four combine the derivatives with
`p`. The final Fourier kernel differentiates and combines those products into
the two explicit field terms. The zonal average is over the padded real `ny`
grid and includes its own `1/ny` factor. Preserve the exact derivative/product
split when optimizing: it is arranged both for dealiasing and buffer reuse.

AB3 is the default timestepper, with RK4 and SSPRK3 inherited from core. AB3
keeps three explicit-term histories in device global memory, while the system
keeps two field time levels. `get_fields(n)` resolves the circular history by
`current_step`; never assume `fields[0]` is always current. The `phi` and `T`
lists are zero-copy CuPy views into each history array. Do not replace or resize
the backing `fields` arrays without rebuilding those views.

## Layout, Fourier convention, and precision

- Arrays are C-contiguous with physical/Fourier axis order `(z, x, y)`, so `y`
  is the contiguous axis. A linear CUDA index is
  `iy + ny * (ix + nx * iz)`.
- Field arrays have shape `(field, nz, nx, half_ny)`, where
  `half_ny = ny // 2 + 1`. Model views have shape `(nz, nx, half_ny)`.
- Real work arrays have shape `(quantity, nz, nx, ny)`. With `nz == 1`, the
  nonlinear kernel may use `index // ny` as `ix`; this is not a valid generic
  3-D indexing shortcut.
- `kx` and `kz` use full FFT order: zero, positive modes, then negative modes.
  `ky` is the nonnegative real-FFT half spectrum. Derivatives use `+i*k`.
- `dimensions.n*` are padded transform sizes. The `*_unpadded` values describe
  the retained solved grid for two-thirds dealiasing. Padded modes must stay
  zero. Phase-shift truncation has different masks and memory modes; use the
  base operation rather than adding a plugin-only FFT path.
- Stored field coefficients use the forward-normalized convention
  `F_k = (1/N) sum_x f_x exp(-ikx)`, `N = nz*nx*ny`. Runtime cuFFT plans are
  unnormalized in both directions. Thus C2R maps stored coefficients directly
  to physical values; R2C nonlinear products are larger than stored
  coefficients by `N`, and `add_nonlinear_terms` applies
  `DFT_FULLSIZE_FACTOR = 1/N`. Do not normalize `dft_bits` earlier as well.
- `cp.fft.irfftn`/`np.fft.irfftn` diagnostic paths use `norm="forward"` to
  reconstruct real fields from the same stored convention.
- Fourier reductions double `iky > 0` to restore the omitted negative-`ky`
  modes. Because stored coefficients are forward-normalized, sums such as
  `sum |T_k|^2` are already box averages by Parseval; do not add another FFT or
  domain-volume factor. Dealiased Nyquist modes are expected to be zero.
- `setup.precision` maps `self.float` to `np.float32`/`np.float64` and
  `self.complex` to `np.complex64`/`np.complex128`. Double precision defines
  `DOUBLE_PRECISION`; CUDA then aliases `FLUCS_FLOAT` and `FLUCS_COMPLEX`
  accordingly. Use these aliases in CUDA and cast scalar kernel arguments with
  `self.float(...)` or `self.int(...)`. NetCDF output follows solver precision.

## CPU/GPU and allocation rules

The timestep hot path is GPU-resident. Fields, FFT intermediates, nonlinear
products, CFL storage, reduction scratch, and raw kernels are CuPy/CUDA data.
Python/NumPy owns configuration, wavenumber metadata, initial/restart data,
linear-reference calculations, diagnostic caches, and netCDF/matplotlib work.
Initial data crosses to the GPU with `.set()`. Calls to `.get()`, `cp.asnumpy`,
or conversion of a CuPy scalar to a Python value transfer to the host and
synchronize the stream.

Preserve these performance choices:

- Allocate persistent GPU memory in setup/`_allocate_memory`, diagnostic
  `init_vars`/`ready`, or the base dealiased-operation factory. Do not allocate
  CuPy arrays inside a timestep or per-mode diagnostic execution without a
  measured need.
- Keep `combine_first_and_second_intermediates=True`. The CUDA product kernel
  reads every aliased input into registers before writing output; changing that
  order can corrupt data. The phase-shift variants may call the same callbacks
  on shifted and unshifted buffers.
- Keep `keep_previous_stage_alive=False` unless new model code truly reads the
  preceding RK stage. It enables core memory reuse.
- Reuse `system.get_temp_array(size, is_complex)` for reduction scratch. It is
  keyed only by size and type, so same-shaped reduction results may alias the
  same device allocation. Consume/copy one result before launching another;
  do not collect several returned scratch views and transfer them later.
- Raw kernels and FFT calls enqueue asynchronously. Avoid explicit
  `synchronize()` in production paths. Transfer only final diagnostic results,
  preferably small reduced data. If several scalars must be transferred in one
  synchronization, first give them distinct persistent device storage or write
  a packed reduction/kernel; simply delaying `.get()` on aliased scratch is
  incorrect.
- Preserve contiguous layout and pass existing arrays/pointers to kernels.
  Prefer fused device functors/reductions over materializing diagnostic arrays.
- Register kernels and reduction templates before `compile_cupy_module()`.
  `KernelWrapper` adds both the CUDA name expression and the wrapper to the
  system collection. Runtime registration normally belongs in `init_vars` or
  `register_kernels`, not in `execute`.

Core compiles the system's adjacent `.cu` through `cp.RawModule`. It adds the
`src` roots for every `FlucsSystem` subclass in the MRO as CUDA include paths,
which is how this plugin can include core headers across repositories. A source
timestamp intentionally defeats CuPy's compiled-module cache. CUDA compilation
at startup is therefore expected. Do not copy core headers into this repo to
avoid that dependency.

## Diagnostics and output

System diagnostics are classes in the `diags` set and are selected by name in
`[output.<name>].diags`. A `FlucsDiagnosticVariable.shape` contains dimension
*names*, while `dimensions` maps those names to coordinate arrays. Use
`save_data()` and let the output layer cache/write values. Complex netCDF data
is stored as `<name>_real` and `<name>_imag`.

The plugin provides:

- `heatflux/heatflux`: `Q = -<T d_y phi>`.
- `free_energy/W`: the temperature-variance invariant
  `W = <T^2>/2` from equation (2.34).
- `free_energy/dWdt`: a one-step backward difference using current and previous
  field histories.
- `free_energy/dWdt_inj = kappaT * Q`.
- `free_energy/dWdt_coll = chi <T laplacian(T)>`, normally negative.
- Per-component hyperdissipation contributions and `dWdt_error`, the residual
  after injection and dissipation are removed from the numerical derivative.
- `zonal_profiles/phi(time, x)` and `zonal_profiles/T(time, x)`: raw zonal
  profiles on the full padded `x` sampling grid. The saved potential retains
  its arbitrary constant mode.
- `zonal_profiles/gamma_max(time)`: the maximum imaginary eigenfrequency over
  the retained solver grid, computed once from `compute_linear_eigensystem()`
  and repeated at diagnostic samples.
- `momentum_flux/Pi_phi = -overline{dx(phi) dy(phi)}` and
  `momentum_flux/Pi_T = -overline{dx(phi) dy(T)}`.
- `momentum_flux/Pi_t = Pi_phi + Pi_T` and
  `momentum_flux/Pi_d = -chi dx^2(coeffa phi - coeffb T)`.
- When `[forcing].method = "alfven_eigenmode"`, `momentum_flux/Pi_AE` records
  a prescribed zonal momentum flux from a single radial Alfvén-mode pair. The
  forcing is applied explicitly to the zonal potential only, and `Pi_total` is
  the sum of `Pi_t`, `Pi_d`, and `Pi_AE`.

The heat-flux and free-energy quantities are device functor reductions over the
rFFT half-grid. The current scalar implementation calls `.get().item()` after
each result. That is required for correctness with shared scalar scratch,
although a future packed diagnostic can reduce synchronizations deliberately.

The zonal-profile diagnostic gathers both `ky=0` field lines into one
persistent contiguous device buffer, transfers them together, and applies a
batched NumPy inverse FFT with `norm="forward"`. The momentum-flux diagnostic
uses `create_dealiased_operation()` for both two-thirds and phase-shift modes:
it transforms `dx(phi)`, `dy(phi)`, and `dy(T)`, forms both products in one
alias-safe CUDA kernel, and gathers only retained `ky=0` output modes. Its
unnormalized nonlinear R2C output receives `DFT_FULLSIZE_FACTOR` during the
gather. `Pi_d` is formed directly in Fourier space with
`COEFFA_TIMES_CHI` and `COEFFB_TIMES_CHI`. All six momentum-flux lines are
transferred together. Keep these buffers and FFT plans persistent; neither
diagnostic should allocate CuPy arrays in `execute()`.

The optional Alfvén-eigenmode forcing is configured with
`forcing.momentum_flux_amplitude`, `forcing.radial_mode_number`,
`forcing.radial_phase`, `forcing.growth_rate`, and `forcing.midpoint_time`.
It is a one-way qualitative model of eigenmode self-interaction, not a coupled
Alfvén-envelope evolution.

Enable both profile diagnostics in a NetCDF output such as:

```toml
[output.1d]
save_steps = 100
type = "netcdf4"
diags = ["zonal_profiles", "momentum_flux"]
```

Diagnostics execute at each output's `save_steps`; disk writes happen at the
global `output.write_steps`. Cached data lives on the CPU between writes, so a
large array diagnostic with frequent saves can consume substantial host memory.
`begin_time_step()` invalidates the shared real-space cache. Multiple
`RealspaceDataDiag` slices without filtering share one transform per step;
preserve that behavior.

Each run/restart appends a numbered top-level group to `output.<name>.nc`. Every
group stores `time`, `dt`, the full input TOML, and diagnostic subgroups.
`FlucsPostProcessing.load_netcdf_variable()` normally concatenates
time-dependent variables and returns `(values, boundary_indices, dims_dicts)`.
Use the boundary indices when plotting restarts and split segments before
differencing or time integration; do not assume samples are continuous or
unique across a boundary. Dimension metadata and even input parameters may
differ by group. For complex data use `load_netcdf_variable_complex()`.

Postprocessing scripts should use the common `FlucsPostProcessing.parser()`,
construct `FlucsPostProcessing` with the exact expected output filenames and an
appropriate solver/system constraint, discover files with
`get_valid_netcdf_paths()`, and save through `post.save()`. Saving is a no-op
unless `--save_directory/-s` is supplied and overwrites by default. Support a
noninteractive matplotlib backend for batch/HPC use.

`plot_zonal_profiles.py` creates a five-panel, time-averaged figure containing
gauge-adjusted zonal potential, zonal flow, zonal shear with
`+/-gamma_max` references, zonal temperature, and its gradient with a
`kappaT` reference. Only the plotted potential has its spatial mean removed.
Flow, shear, and temperature gradients are spectral derivatives of the
time-averaged periodic profiles. `kappaT` comes from the input TOML embedded in
the contributing NetCDF group.

`plot_momentum_flux.py` creates a momentum-flux decomposition panel and a
balance panel for each simulation. It warns when generic hyperdissipation is
active because `Pi_t + Pi_d` omits that contribution. Both profile scripts
honour `--groups`, accept either `--time START:END` or `--fraction` (never
both), and default to the final 20% when neither is supplied. Time bounds are
inclusive. They integrate each output group separately with trapezoidal
physical-time weighting, combine integrals and durations, allow one shared
restart endpoint, and reject positive-duration overlap. Preserve validation
of monotonic times, at least two samples and positive selected duration,
compatible `x` grids, finite data, and consistent `gamma_max`, `kappaT`,
`chi`, `coeffa`, and `coeffb` across contributing groups.

Known diagnostic/postprocessing sharp edges:

- `dWdt` has no physically meaningful previous sample at the initial output;
  conservation plotting intentionally omits index 0 from its error curve.
- The legacy heat-flux and conservation scripts inherit `--groups` but do not
  pass `args.groups` to their load calls, so they always load all groups. The
  zonal-profile and momentum-flux scripts do support selected groups.
- `plot_heatflux_vs_time.py` takes x-limits from the last loaded simulation,
  which is wrong for runs with different time spans, and an empty match leads
  to an unbound `time` variable. Address these in targeted postprocessing work.
- `check_conservation_energy.py` recomputes its balance residual rather than
  reading `free_energy/dWdt_error`; keep the two definitions consistent.
- Avoid numerical derivatives or rolling windows across netCDF group
  boundaries. Treat missing variables (filled with NaN by the loader) before
  statistics.

## Change and validation guidance

Keep solver changes narrow and compare Python and CUDA expressions term by
term. Any new field, product, or kernel changes compile-time counts, allocation
shapes, callback indexing, and often timestepper interfaces together. Any
change to core APIs may require a companion change in the `flucs` checkout;
do not add a local workaround until inspecting the core implementation.

There is currently no bundled run input. At minimum:

1. Run `python -m compileall -q src/flucs_fluid_itg`.
2. Run Ruff on touched Python files. The core repo uses an 80-column Ruff
   configuration; this plugin currently has some pre-existing lint debt, so do
   not mix unrelated cleanup into numerical changes. Put the Ruff cache under
   `/tmp` if the repository root is not writable.
3. In a CUDA-capable environment with both editable installs, run `flucs
   --list`, then a short representative case with linear-matrix checking left
   enabled. A CPU-only import/compile check cannot compile RawModule kernels or
   validate the numerical path.
4. For diagnostic changes, inspect the resulting netCDF group/variable shapes,
   values, restart boundaries, and both single- and double-precision behavior.
   For postprocessing, exercise one directory, multiple directories, selected
   groups, missing variables, and headless figure saving as applicable.

Do not silently alter physical normalization, signs, zero/zonal-mode handling,
or diagnostic definitions to make a plot look plausible. Derive the expected
quantity from the paper, verify the framework's transform/reduction convention,
and record any intentional definition change.
