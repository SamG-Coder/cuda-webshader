/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Constants from NVIDIA dwtHaar1D.cu.
#define INV_SQRT_2 0.70710678118654752440f
#define LOG_NUM_BANKS 4
namespace cg = cooperative_groups;

__global__ void dwtHaar1D(float             *id,
                          float             *od,
                          float             *approx_final,
                          const unsigned int dlevels,
                          const unsigned int slength_step_half,
                          const int          bdim)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // shared memory for part of the signal
    extern __shared__ float shared[];

    // thread runtime environment, 1D parametrization
    const int gdim = gridDim.x;
    // const int bdim = blockDim.x;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    // global thread id (w.r.t. to total data set)
    const int    tid_global = (bid * bdim) + tid;
    unsigned int idata      = (bid * (2 * bdim)) + tid;

    // read data from global memory
    shared[tid]        = id[idata];
    shared[tid + bdim] = id[idata + bdim];
    cg::sync(cta);

    // this operation has a two way bank conflicts for all threads, this are two
    // additional cycles for each warp -- all alternatives to avoid this bank
    // conflict are more expensive than the one cycle introduced by serialization
    float data0 = shared[2 * tid];
    float data1 = shared[(2 * tid) + 1];
    cg::sync(cta);

    // detail coefficient, not further referenced so directly store in
    // global memory
    od[tid_global + slength_step_half] = (data0 - data1) * INV_SQRT_2;

    // offset to avoid bank conflicts
    // see the scan example for a more detailed description
    unsigned int atid = tid + (tid >> LOG_NUM_BANKS);

    // approximation coefficient
    // store in shared memory for further decomposition steps in this global step
    shared[atid] = (data0 + data1) * INV_SQRT_2;

    // all threads have to write approximation coefficient to shared memory before
    // next steps can take place
    cg::sync(cta);

    // early out if possible
    // the compiler removes this part from the source because dlevels is
    // a constant shader input
    // note: syncthreads in bodies of branches can lead to dead-locks unless
    // the condition evaluates the same way for ALL threads of a block, as in
    // this case
    if (dlevels > 1) {
        // offset to second element in shared element which has to be used for the
        // decomposition, effectively 2^(i - 1)
        unsigned int offset_neighbor = 1;
        // number of active threads per decomposition level
        // identical to the offset for the detail coefficients
        unsigned int num_threads = bdim >> 1;

        // index for the first element of the pair to process
        // the representation is still compact (and therefore still tid * 2)
        // because the first step operated on registers and only the result has been
        // written to shared memory
        unsigned int idata0 = tid * 2;

        // offset levels to make the loop more efficient
        for (unsigned int i = 1; i < dlevels; ++i) {
            // Non-coalesced writes occur if the number of active threads becomes
            // less than 16 for a block because the start address for the first
            // block is not always aligned with 64 byte which is necessary for
            // coalesced access. However, the problem only occurs at high levels
            // with only a small number of active threads so that the total number of
            // non-coalesced access is rather small and does not justify the
            // computations which are necessary to avoid these uncoalesced writes
            // (this has been tested and verified)
            if (tid < num_threads) {
                // update stride, with each decomposition level the stride grows by a
                // factor of 2
                unsigned int idata1 = idata0 + offset_neighbor;

                // position of write into global memory
                unsigned int g_wpos = (num_threads * gdim) + (bid * num_threads) + tid;

                // compute wavelet decomposition step

                // offset to avoid bank conflicts
                unsigned int c_idata0 = idata0 + (idata0 >> LOG_NUM_BANKS);
                unsigned int c_idata1 = idata1 + (idata1 >> LOG_NUM_BANKS);

                // detail coefficient, not further modified so directly store
                // in global memory
                od[g_wpos] = (shared[c_idata0] - shared[c_idata1]) * INV_SQRT_2;

                // approximation coefficient
                // note that the representation in shared memory becomes rather sparse
                // (with a lot of holes inbetween) but the storing scheme in global
                // memory guarantees that the common representation (approx, detail_0,
                // detail_1, ...)
                // is achieved
                shared[c_idata0] = (shared[c_idata0] + shared[c_idata1]) * INV_SQRT_2;

                // update storage offset for details
                num_threads = num_threads >> 1; // div 2
                offset_neighbor <<= 1;          // mul 2
                idata0 = idata0 << 1;           // mul 2
            }

            // sync after each decomposition step
            cg::sync(cta);
        }

        // write the top most level element for the next decomposition steps
        // which are performed after an interlock synchronization on host side
        if (0 == tid) {
            approx_final[bid] = shared[0];
        }

    } // end early out if possible
}
