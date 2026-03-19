/*
 * Contains all the CUDA kernels for the 2D ITG model of Ivanov et al. (2020).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"

extern "C" {

// Array for AB3 nonlinear terms
__constant__ FLUCS_COMPLEX* multistep_nonlinear_terms = NULL;

__device__ void get_linear_matrix(const int index, const FLUCS_FLOAT dt, FLUCS_COMPLEX matrix[2][2]){
    // First, we need to figure out the kx and ky of the mode.
    // const int ikx = index / HALF_NY;
    // const int iky = index % HALF_NY;

    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const int ikx = indices.ikx;
    const int iky = indices.iky;

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * iky;
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky + (FLUCS_FLOAT)(index == 0);
    const FLUCS_FLOAT eta_inv = (FLUCS_FLOAT)(1.0) / ((FLUCS_FLOAT)(iky > 0) + kperp2);


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


__global__ void find_derivatives(const FLUCS_COMPLEX* fields,
                                 FLUCS_COMPLEX* dft_derivatives,
                                 FLUCS_FLOAT* real_dxphi_zonal,
                                 FLUCS_FLOAT* cfl_rate){
    const int padded_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(padded_index < HALFPADDEDSIZE))
        return;


    // const int padded_ikx = padded_index / HALF_PADDED_NY;
    // const int padded_iky = padded_index % HALF_PADDED_NY;
    //
    indices3d_t padded_indices = get_indices3d<1, PADDED_NX, HALF_PADDED_NY>(padded_index);
    const int padded_ikx = padded_indices.padded_ikx;
    const int padded_iky = padded_indices.padded_iky;

    // Use this kernel to also zero out real_dxphi_zonal and cfl_rate
    if (padded_iky == 0)
        real_dxphi_zonal[padded_ikx] = 0;

    if (padded_index == 0)
        cfl_rate[0] = 0;

    // Check if mode should be zeroed
    if ((padded_ikx >= HALF_NX && padded_ikx < HALF_NX - NX + PADDED_NX)
        || padded_iky >= HALF_NY){

        dft_derivatives[padded_index] = 0;
        dft_derivatives[padded_index + HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 2*HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 3*HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 4*HALFPADDEDSIZE] = 0;
        return;
    }
    
    // const int ikx = (padded_ikx < HALF_NX) ? padded_ikx : NX - PADDED_NX + padded_ikx;
    const int ikx = ikx_from_padded_ikx(padded_ikx);

    // padded_iky and iky are the same for nonzero modes
    // const int index = padded_iky + HALF_NY * ikx;
    const int index = index_from_3d<1, NX, HALF_NY>(0, ikx, padded_iky);

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * padded_iky;
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(padded_iky);

    const FLUCS_FLOAT ky2minuskx2 = ky*ky - kx*kx;
    const FLUCS_FLOAT minus_kxky = -kx*ky;

    const FLUCS_COMPLEX phi = fields[index];
    const FLUCS_COMPLEX T = fields[index + HALFUNPADDEDSIZE];

    dft_derivatives[padded_index]\
        = FLUCS_COMPLEX(-kx * phi.imag(), kx * phi.real());

    dft_derivatives[padded_index + HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-ky * phi.imag(), ky * phi.real());

    dft_derivatives[padded_index + 2*HALFPADDEDSIZE]\
        = ky2minuskx2 * phi;

    dft_derivatives[padded_index + 3*HALFPADDEDSIZE]\
        = minus_kxky * phi;

    dft_derivatives[padded_index + 4*HALFPADDEDSIZE]\
        = phi + T;
}


__global__ void find_nonlinear_bits(FLUCS_FLOAT* real_derivatives_and_bits,
                                    const FLUCS_FLOAT* real_dxphi_zonal,
                                    FLUCS_FLOAT* cfl_rate){
    // Shared memory for CFL calculations
    extern __shared__ float cfl_shared[];

    const int real_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(real_index < PADDEDSIZE))
        return;

    const FLUCS_FLOAT cfl = flucs_fabs(real_derivatives_and_bits[real_index]) * (NY / LY) + flucs_fabs(real_derivatives_and_bits[real_index + PADDEDSIZE]) * (NX / LX);
    // cfl_array[real_index] = cfl;

    // Find max CFL using shared memory
    // TODO: Could we speed this up by reducing over warps?
    cfl_shared[threadIdx.x] = cfl;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            cfl_shared[threadIdx.x] = fmaxf(cfl_shared[threadIdx.x], cfl_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // First thread in block writes to global max via atomic
    if (threadIdx.x == 0) {
        atomicMaxFloat(cfl_rate, cfl_shared[0]); // custom atomic for float
    }

    // index inside the zonal phi array
    const int ix = real_index / PADDED_NY;

    const FLUCS_FLOAT dxphi = real_derivatives_and_bits[real_index];
    const FLUCS_FLOAT dyphi = real_derivatives_and_bits[real_index + PADDEDSIZE];
    const FLUCS_FLOAT dx2mdy2phi = real_derivatives_and_bits[real_index + 2*PADDEDSIZE];
    const FLUCS_FLOAT dxdyphi = real_derivatives_and_bits[real_index + 3*PADDEDSIZE];
    const FLUCS_FLOAT p = real_derivatives_and_bits[real_index + 4*PADDEDSIZE];

    // dxphi_zonal * dyphi
    real_derivatives_and_bits[real_index] = real_dxphi_zonal[ix] * dyphi;

    // (dx^2 - dy^2)phi * p
    real_derivatives_and_bits[real_index + PADDEDSIZE] = dx2mdy2phi * p;
                                        
    // dxdyphi p
    real_derivatives_and_bits[real_index + 2*PADDEDSIZE] = dxdyphi * p;
 
    // dxphi p
    real_derivatives_and_bits[real_index + 3*PADDEDSIZE] = dxphi * p;

    // dyphi p
    real_derivatives_and_bits[real_index + 4*PADDEDSIZE] = dyphi * p;
}

__device__ void add_nonlinear_terms(const int index,
                                    const FLUCS_FLOAT dt,
                                    const int current_step,
                                    const FLUCS_FLOAT AB0,
                                    const FLUCS_FLOAT AB1,
                                    const FLUCS_FLOAT AB2,
                                    const FLUCS_COMPLEX* dft_bits,
                                    FLUCS_COMPLEX* rhs_fields){
    // const int ikx = index / HALF_NY;
    // const int iky = index % HALF_NY;

    indices3d_t indices = get_indices3d<1, NX, HALF_NY>(index);
    const int ikx = indices.ikx;
    const int iky = indices.iky;

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * iky;
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    // const int ikx_padded = (ikx < HALF_NX) ? ikx : PADDED_NX - NX + ikx;
    const int padded_ikx = padded_ikx_from_ikx(ikx);

    // const int padded_index = HALF_PADDED_NY * padded_ikx + iky;
    const int padded_index = index_from_3d<1, PADDED_NX, HALF_PADDED_NY>(0, padded_ikx, iky);


    const FLUCS_FLOAT kx2mky2 = kx*kx - ky*ky;
    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky + (FLUCS_FLOAT)(index == 0);
    const FLUCS_FLOAT eta_inv = (FLUCS_FLOAT)(1.0) / ((FLUCS_FLOAT)(iky > 0) + kperp2);
    
    const FLUCS_COMPLEX phiNL = DFT_PADDEDSIZE_FACTOR * \
        eta_inv*(dft_bits[padded_index]
                 - kx*ky*dft_bits[padded_index + HALFPADDEDSIZE]
                 + kx2mky2*dft_bits[padded_index + 2*HALFPADDEDSIZE]);

    const FLUCS_COMPLEX TNL = DFT_PADDEDSIZE_FACTOR * (
                              FLUCS_COMPLEX(-ky * dft_bits[padded_index + 3*HALFPADDEDSIZE].imag(),
                                             ky * dft_bits[padded_index + 3*HALFPADDEDSIZE].real())
                             +FLUCS_COMPLEX( kx * dft_bits[padded_index + 4*HALFPADDEDSIZE].imag(),
                                            -kx * dft_bits[padded_index + 4*HALFPADDEDSIZE].real()));

    const int multistep_index_0 = ((current_step      % 3 + 3) % 3) * 2 * HALFUNPADDEDSIZE + index;
    const int multistep_index_1 = ((current_step + 2) % 3)          * 2 * HALFUNPADDEDSIZE + index;
    const int multistep_index_2 = ((current_step + 1) % 3)          * 2 * HALFUNPADDEDSIZE + index;

    // phi
    rhs_fields[0] -= dt * (AB0*phiNL
                           +AB1*multistep_nonlinear_terms[multistep_index_1]
                           +AB2*multistep_nonlinear_terms[multistep_index_2]);

    multistep_nonlinear_terms[multistep_index_0] = phiNL;

    // T
    rhs_fields[1] -= dt * (AB0*TNL
                           +AB1*multistep_nonlinear_terms[multistep_index_1 + HALFUNPADDEDSIZE]
                           +AB2*multistep_nonlinear_terms[multistep_index_2 + HALFUNPADDEDSIZE]);

    multistep_nonlinear_terms[multistep_index_0 + HALFUNPADDEDSIZE] = TNL;
}

__global__
void heatflux_kx(
    const FLUCS_COMPLEX* phi,
    const FLUCS_COMPLEX* T,
    FLUCS_COMPLEX* output){

    multiply_and_sum_last_axis<HALF_NY, true>(
            COMPLEX_ONE,
            output,
            Dy_Functor{phi},
            CC_Functor{T}
        );

}

__global__
void dW_kx(
    const FLUCS_COMPLEX* T_now,
    const FLUCS_COMPLEX* T_prev,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            (FLUCS_FLOAT)0.5,
            output,
            Abs2_Functor{T_now, FLOAT_ONE},
            Abs2_Functor{T_prev, -FLOAT_ONE}
        );

}

__global__
void free_energy_kx(
    const FLUCS_COMPLEX* T,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            (FLUCS_FLOAT)0.5,
            output,
            Abs2_Functor{T, FLOAT_ONE}
        );

}

__global__
void free_energy_collisional_loss_kx(
    const FLUCS_COMPLEX* T,
    FLUCS_COMPLEX* output){

    multiply_and_sum_last_axis<HALF_NY, true>(
            FLUCS_COMPLEX(CHI, 0),
            output,
            DelPerp2_Functor{T},
            CC_Functor{T}
        );

}

} // extern "C"
