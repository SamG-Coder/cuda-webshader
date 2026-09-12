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

#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)
// Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////
static __constant__ unsigned int c_Table[QRNG_DIMENSIONS][QRNG_RESOLUTION];

static __global__ void quasirandomGeneratorKernel(float *d_Output, unsigned int seed, unsigned int N)
{
    unsigned int *dimBase = &c_Table[threadIdx.y][0];
    unsigned int  tid     = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);

    for (unsigned int pos = tid; pos < N; pos += threadN) {
        unsigned int result = 0;
        unsigned int data   = seed + pos;

        for (int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if (data & 1) {
                result ^= dimBase[bit];
            }

        d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
    }
}


// MIT display helper: pack the three original coordinate planes as float4.
// Coordinates are copied without scaling, resampling or CPU generation.
__global__ void packQuasirandomPoints(const float* values,float4* positions,unsigned int N){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<N)positions[i]=make_float4(values[i],values[N+i],values[2*N+i],1.0f);
}
