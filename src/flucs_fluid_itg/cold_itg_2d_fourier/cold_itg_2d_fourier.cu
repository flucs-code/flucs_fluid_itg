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
    // First, we need to figure out the kx and ky of the mode.
    // const size_t ikx = index / HALF_NY;
    // const size_t iky = index % HALF_NY;

    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * iky;
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    const FLUCS_FLOAT eta_inv = (FLUCS_FLOAT)(1.0) / ((FLUCS_FLOAT)(iky > 0) + kperp2 + (FLUCS_FLOAT)(index == 0));


    // Generate the linear matrix
    // const FLUCS_COMPLEX matrix_phiphi = FLUCS_COMPLEX(
    matrix[0][0] = FLUCS_COMPLEX(
        A_TIMES_CHI*kperp2*kperp2,
        -ky*(KAPPA_B - KAPPA_N) - KAPPA_T*kperp2*ky) * eta_inv;

    // const FLUCS_COMPLEX matrix_phiT = FLUCS_COMPLEX(
    matrix[0][1] = FLUCS_COMPLEX(
        -B_TIMES_CHI*kperp2*kperp2,
        -ky*KAPPA_B) * eta_inv;

    // const FLUCS_COMPLEX matrix_Tphi = FLUCS_COMPLEX(
    matrix[1][0] = FLUCS_COMPLEX(
        0,
        KAPPA_T*ky);

    // const FLUCS_COMPLEX matrix_TT = FLUCS_COMPLEX(
    matrix[1][1] = FLUCS_COMPLEX(
        CHI*kperp2,
        0);
}


__global__ void find_derivatives(
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
    FLUCS_COMPLEX dft_derivatives_global[NUMBER_OF_DFT_DERIVATIVES][HALFPADDEDSIZE],
    FLUCS_FLOAT real_dxphi_zonal_global[PADDED_NX])
{
    const size_t padded_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(padded_index < HALFPADDEDSIZE))
        return;


    // const size_t padded_ikx = padded_index / HALF_PADDED_NY;
    // const size_t padded_iky = padded_index % HALF_PADDED_NY;
    //
    indices3d_t padded_indices = get_indices3d<1, PADDED_NX, HALF_PADDED_NY>(padded_index);
    const size_t padded_ikx = padded_indices.padded_ikx;
    const size_t padded_iky = padded_indices.padded_iky;

    // Use this kernel to also zero out real_dxphi_zonal and cfl_rate
    if (padded_iky == 0)
        real_dxphi_zonal_global[padded_ikx] = 0;

    // Check if mode should be zeroed
    if ((padded_ikx >= HALF_NX && padded_ikx < HALF_NX - NX + PADDED_NX)
        || padded_iky >= HALF_NY){

        dft_derivatives_global[0][padded_index] = 0;
        dft_derivatives_global[1][padded_index] = 0;
        dft_derivatives_global[2][padded_index] = 0;
        dft_derivatives_global[3][padded_index] = 0;
        dft_derivatives_global[4][padded_index] = 0;
        return;
    }
    
    // const size_t ikx = (padded_ikx < HALF_NX) ? padded_ikx : NX - PADDED_NX + padded_ikx;
    const size_t ikx = ikx_from_padded_ikx(padded_ikx);

    // padded_iky and iky are the same for nonzero modes
    // const size_t index = padded_iky + HALF_NY * ikx;
    const size_t index = index_from_3d<1, NX, HALF_NY>(0, ikx, padded_iky);

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * padded_iky;
    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(padded_iky);

    const FLUCS_FLOAT dx2minusdy2 = (dx*dx - dy*dy).real();
    const FLUCS_FLOAT dxdy = (dx*dy).real();

    const FLUCS_COMPLEX phi = fields_global[0][index];
    const FLUCS_COMPLEX T = fields_global[1][index];

    dft_derivatives_global[0][padded_index] = dx * phi;

    dft_derivatives_global[1][padded_index] = dy * phi;

    dft_derivatives_global[2][padded_index] = dx2minusdy2 * phi;

    dft_derivatives_global[3][padded_index] = dxdy * phi;

    dft_derivatives_global[4][padded_index] = phi + T;
}


__global__ void find_nonlinear_bits(FLUCS_FLOAT real_derivatives_and_bits_global[NUMBER_OF_DFT_COMBINED][PADDEDSIZE],
                                    const FLUCS_FLOAT real_dxphi_zonal_global[PADDED_NX],
                                    FLUCS_FLOAT* cfl_rate_global){
    const size_t real_index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = real_index < PADDEDSIZE;

    // Inactive threads do not contribute to the cfl reduction 
    const FLUCS_FLOAT dxphi = in_bounds
        ? real_derivatives_and_bits_global[0][real_index]
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT dyphi = in_bounds
        ? real_derivatives_and_bits_global[1][real_index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT cfl = flucs_fabs(dxphi) * (NY / LY)
        + flucs_fabs(dyphi) * (NX / LX);

    update_cfl(cfl, cfl_rate_global);

    // Out-of-bounds threads should not contribute to nonlinear bits
    if (!in_bounds)
        return;

    // index inside the zonal phi array
    const size_t ix = real_index / PADDED_NY;

    const FLUCS_FLOAT dx2mdy2phi = real_derivatives_and_bits_global[2][real_index];
    const FLUCS_FLOAT dxdyphi = real_derivatives_and_bits_global[3][real_index];
    const FLUCS_FLOAT p = real_derivatives_and_bits_global[4][real_index];

    // dxphi_zonal * dyphi
    real_derivatives_and_bits_global[0][real_index] = real_dxphi_zonal_global[ix] * dyphi;

    // (dx^2 - dy^2)phi * p
    real_derivatives_and_bits_global[1][real_index] = dx2mdy2phi * p;
                                        
    // dxdyphi p
    real_derivatives_and_bits_global[2][real_index] = dxdyphi * p;
 
    // dxphi p
    real_derivatives_and_bits_global[3][real_index] = dxphi * p;

    // dyphi p
    real_derivatives_and_bits_global[4][real_index] = dyphi * p;
}

__device__ void add_nonlinear_terms(const size_t index,
                                    const FLUCS_FLOAT dt,
                                    const FLUCS_FLOAT current_time,
                                    const long long current_step,
                                    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
                                    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]) {

    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * iky;
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    // const size_t ikx_padded = (ikx < HALF_NX) ? ikx : PADDED_NX - NX + ikx;
    const size_t padded_ikx = padded_ikx_from_ikx(ikx);

    // const size_t padded_index = HALF_PADDED_NY * padded_ikx + iky;
    const size_t padded_index = index_from_3d<1, PADDED_NX, HALF_PADDED_NY>(0, padded_ikx, iky);


    const FLUCS_FLOAT kx2mky2 = kx*kx - ky*ky;
    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky + (FLUCS_FLOAT)(index == 0);
    const FLUCS_FLOAT eta_inv = (FLUCS_FLOAT)(1.0) / ((FLUCS_FLOAT)(iky > 0) + kperp2);


    const FLUCS_COMPLEX dxphi0_dyphi = dft_bits_global[0][padded_index];
    const FLUCS_COMPLEX dx2mdy2phi_p = dft_bits_global[1][padded_index];
    const FLUCS_COMPLEX dxdyphi_p    = dft_bits_global[2][padded_index];
    const FLUCS_COMPLEX dxphi_p      = dft_bits_global[3][padded_index];
    const FLUCS_COMPLEX dyphi_p      = dft_bits_global[4][padded_index];
    
    explicit_terms[0] += DFT_PADDEDSIZE_FACTOR * \
        eta_inv*(dxphi0_dyphi
                 - kx*ky * dx2mdy2phi_p
                 + kx2mky2 * dxdyphi_p);

    explicit_terms[1] += DFT_PADDEDSIZE_FACTOR * (dy * dxphi_p - dx * dyphi_p);
}

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* T_global = fields_global + HALFUNPADDEDSIZE;
        return Abs2_Functor{T_global, 0.5}(index);
    }
};

struct FreeEnergyColl_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* T_global = fields_global + HALFUNPADDEDSIZE;
        return CHI*(DelPerp2_Functor{T_global}(index) * CC_Functor{T_global}(index)).real();
    }
};


struct Heatflux_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* phi_global = fields_global;
        const FLUCS_COMPLEX* T_global = fields_global + HALFUNPADDEDSIZE;

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
