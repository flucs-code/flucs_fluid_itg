/*
 * Contains all the CUDA kernels for the 2D ITG model of Ivanov et al. (2020).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"

extern "C" {

__device__ void get_linear_matrix(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    FLUCS_COMPLEX matrix[2][2]
) {
    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    const FLUCS_FLOAT eta_inv = (FLUCS_FLOAT)(1.0) / ((FLUCS_FLOAT)(iky > 0) + kperp2 + (FLUCS_FLOAT)(index == 0));


    // Generate the linear matrix
    matrix[0][0] = FLUCS_COMPLEX(
        A_TIMES_CHI*kperp2*kperp2,
        -ky*(KAPPA_B - KAPPA_N) - KAPPA_T*kperp2*ky
    ) * eta_inv;

    matrix[0][1] = FLUCS_COMPLEX(
        -B_TIMES_CHI*kperp2*kperp2, 
        -ky*KAPPA_B
    ) * eta_inv;

    matrix[1][0] = FLUCS_COMPLEX(0, KAPPA_T*ky);
    matrix[1][1] = FLUCS_COMPLEX(CHI*kperp2, 0);
}


__global__ void find_derivatives(
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX dft_derivatives_global[NUMBER_OF_DFT_DERIVATIVES][HALFSIZE]
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if index is within bounds
    if (!(index < HALFSIZE))
        return;

    // Zero out padded modes
    if (is_mode_padded(index)) {
        dft_derivatives_global[0][index] = 0;
        dft_derivatives_global[1][index] = 0;
        dft_derivatives_global[2][index] = 0;
        dft_derivatives_global[3][index] = 0;
        dft_derivatives_global[4][index] = 0;
        return;
    }

    // Indices
    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Derivative operations
    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    const FLUCS_FLOAT dx2minusdy2 = (dx * dx - dy * dy).real();
    const FLUCS_FLOAT dxdy = (dx * dy).real();

    // Fields
    const FLUCS_COMPLEX phi = fields_global[0][index];
    const FLUCS_COMPLEX T = fields_global[1][index];

    dft_derivatives_global[0][index] = dx * phi;
    dft_derivatives_global[1][index] = dy * phi;
    dft_derivatives_global[2][index] = dx2minusdy2 * phi;
    dft_derivatives_global[3][index] = dxdy * phi;
    dft_derivatives_global[4][index] = phi + T;
}


__global__ void find_nonlinear_bits(
    const FLUCS_FLOAT real_derivatives_global
        [NUMBER_OF_DFT_DERIVATIVES][FULLSIZE],
    FLUCS_FLOAT real_bits_global[NUMBER_OF_DFT_BITS][FULLSIZE],
    const FLUCS_FLOAT real_dxphi_zonal_global[NX],
    const bool calculate_cfl,
    FLUCS_FLOAT* cfl_rate_global
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = index < FULLSIZE;

    // Ensure that the entire block is handled correctly
    const FLUCS_FLOAT dxphi = in_bounds
        ? real_derivatives_global[0][index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT dyphi = in_bounds
        ? real_derivatives_global[1][index]
        : (FLUCS_FLOAT)0;

    if (calculate_cfl) {
        const FLUCS_FLOAT cfl_rate =
              flucs_fabs(dxphi) * (NY_UNPADDED / LY)
            + flucs_fabs(dyphi) * (NX_UNPADDED / LX);
        update_cfl(cfl_rate, cfl_rate_global);
    }

    if (!in_bounds)
        return;

    // Since NZ == 1, index / NY is the real-space x index.
    const size_t ix = index / NY;

    // Read all inputs before writing because the derivative and product
    // arrays may alias when combine_derivatives_and_bits is enabled.
    const FLUCS_FLOAT dx2mdy2phi = real_derivatives_global[2][index];
    const FLUCS_FLOAT dxdyphi = real_derivatives_global[3][index];
    const FLUCS_FLOAT p = real_derivatives_global[4][index];

    // Calculate real bits
    real_bits_global[0][index] = real_dxphi_zonal_global[ix] * dyphi;
    real_bits_global[1][index] = dx2mdy2phi * p;
    real_bits_global[2][index] = dxdyphi * p;
    real_bits_global[3][index] = dxphi * p;
    real_bits_global[4][index] = dyphi * p;
}

__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
) {
    // Indices
    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Derivative operations
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    const FLUCS_FLOAT kx2mky2 = kx * kx - ky * ky;
    const FLUCS_FLOAT kperp2 =
        kx * kx + ky * ky + (FLUCS_FLOAT)(index == 0);
    const FLUCS_FLOAT eta_inv =
        (FLUCS_FLOAT)1.0 / ((FLUCS_FLOAT)(iky > 0) + kperp2);

    // Extract nonlinear terms from dft_bits_global
    const FLUCS_COMPLEX dxphi0_dyphi = dft_bits_global[0][index];
    const FLUCS_COMPLEX dx2mdy2phi_p = dft_bits_global[1][index];
    const FLUCS_COMPLEX dxdyphi_p = dft_bits_global[2][index];
    const FLUCS_COMPLEX dxphi_p = dft_bits_global[3][index];
    const FLUCS_COMPLEX dyphi_p = dft_bits_global[4][index];

    // Add nonlinear terms
    explicit_terms[0] += DFT_FULLSIZE_FACTOR * eta_inv * (
        dxphi0_dyphi
        - kx * ky * dx2mdy2phi_p
        + kx2mky2 * dxdyphi_p
    );

    explicit_terms[1] += DFT_FULLSIZE_FACTOR * (
        dy * dxphi_p - dx * dyphi_p
    );
}

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* T_global = fields_global + HALFSIZE;
        return Abs2_Functor{T_global, 0.5}(index);
    }
};

struct FreeEnergyColl_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* T_global = fields_global + HALFSIZE;
        return CHI*(DelPerp2_Functor{T_global}(index) * CC_Functor{T_global}(index)).real();
    }
};


struct Heatflux_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* phi_global = fields_global;
        const FLUCS_COMPLEX* T_global = fields_global + HALFSIZE;

        return -(Dy_Functor{phi_global}(index) * CC_Functor{T_global}(index)).real();
    }
};

struct FreeEnergyHyperdissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};


} // extern "C"
