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

namespace cg = cooperative_groups;
__global__ void fwtBatch1Kernel(float *d_Output, float *d_Input, int log2N)
{
    // Handle to thread block group
    cg::thread_block cta  = cg::this_thread_block();
    const int        N    = 1 << log2N;
    const int        base = blockIdx.x << log2N;

    //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ float s_data[];
    float                  *d_Src = d_Input + base;
    float                  *d_Dst = d_Output + base;

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
        s_data[pos] = d_Src[pos];
    }

    // Main radix-4 stages
    const int pos = threadIdx.x;

    for (int stride = N >> 2; stride > 0; stride >>= 2) {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        cg::sync(cta);
        float D0 = s_data[i0];
        float D1 = s_data[i1];
        float D2 = s_data[i2];
        float D3 = s_data[i3];

        float T;
        T          = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T          = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T          = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T          = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
    }

    // Do single radix-2 stage for odd power of two
    if (log2N & 1) {
        cg::sync(cta);

        for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            float D0   = s_data[i0];
            float D1   = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    cg::sync(cta);

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
        d_Dst[pos] = s_data[pos];
    }
}
