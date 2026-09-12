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
// Original NVIDIA integration kernel; host includes isolated for compilation.
__global__ void
d_integrate_trapezoidal(cudaExtent extent, cudaTextureObject_t transferTex, cudaSurfaceObject_t transferIntegrateSurf)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;

    // for higher speed could use hierarchical approach for sum
    if (x >= extent.width) {
        return;
    }

    float stepsize = 1.0 / float(extent.width - 1);
    float to       = float(x) * stepsize;

    float4 outclr = make_float4(0, 0, 0, 0);
    float  incr   = stepsize;

    float4 lastval = tex1D<float4>(transferTex, 0);

    float cur = incr;

    while (cur < to + incr * 0.5) {
        float4 val       = tex1D<float4>(transferTex, cur);
        float4 trapezoid = (lastval + val) / 2.0f;
        lastval          = val;

        outclr += trapezoid;
        cur += incr;
    }

    // surface writes need byte offsets for x!
    surf1Dwrite(outclr, transferIntegrateSurf, x * sizeof(float4));
}

