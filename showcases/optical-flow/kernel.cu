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

__global__ void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= count)
        return;

    sum[pos] = op1[pos] + op2[pos];
}

__global__ void ComputeDerivativesKernel(int                 width,
                                         int                 height,
                                         int                 stride,
                                         float              *Ix,
                                         float              *Iy,
                                         float              *Iz,
                                         cudaTextureObject_t texSource,
                                         cudaTextureObject_t texTarget)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    float t0, t1;
    // x derivative
    t0 = tex2D<float>(texSource, x - 2.0f * dx, y);
    t0 -= tex2D<float>(texSource, x - 1.0f * dx, y) * 8.0f;
    t0 += tex2D<float>(texSource, x + 1.0f * dx, y) * 8.0f;
    t0 -= tex2D<float>(texSource, x + 2.0f * dx, y);
    t0 /= 12.0f;

    t1 = tex2D<float>(texTarget, x - 2.0f * dx, y);
    t1 -= tex2D<float>(texTarget, x - 1.0f * dx, y) * 8.0f;
    t1 += tex2D<float>(texTarget, x + 1.0f * dx, y) * 8.0f;
    t1 -= tex2D<float>(texTarget, x + 2.0f * dx, y);
    t1 /= 12.0f;

    Ix[pos] = (t0 + t1) * 0.5f;

    // t derivative
    Iz[pos] = tex2D<float>(texTarget, x, y) - tex2D<float>(texSource, x, y);

    // y derivative
    t0 = tex2D<float>(texSource, x, y - 2.0f * dy);
    t0 -= tex2D<float>(texSource, x, y - 1.0f * dy) * 8.0f;
    t0 += tex2D<float>(texSource, x, y + 1.0f * dy) * 8.0f;
    t0 -= tex2D<float>(texSource, x, y + 2.0f * dy);
    t0 /= 12.0f;

    t1 = tex2D<float>(texTarget, x, y - 2.0f * dy);
    t1 -= tex2D<float>(texTarget, x, y - 1.0f * dy) * 8.0f;
    t1 += tex2D<float>(texTarget, x, y + 1.0f * dy) * 8.0f;
    t1 -= tex2D<float>(texTarget, x, y + 2.0f * dy);
    t1 /= 12.0f;

    Iy[pos] = (t0 + t1) * 0.5f;
}

__global__ void DownscaleKernel(int width, int height, int stride, float *out, cudaTextureObject_t texFine)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) {
        return;
    }

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    out[ix + iy * stride] = 0.25f
                          * (tex2D<float>(texFine, x - dx * 0.25f, y) + tex2D<float>(texFine, x + dx * 0.25f, y)
                             + tex2D<float>(texFine, x, y - dy * 0.25f) + tex2D<float>(texFine, x, y + dy * 0.25f));
}

template <int bx, int by>
__global__ void JacobiIteration(const float *du0,
                                const float *dv0,
                                const float *Ix,
                                const float *Iy,
                                const float *Iz,
                                int          w,
                                int          h,
                                int          s,
                                float        alpha,
                                float       *du1,
                                float       *dv1)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    volatile __shared__ float du[(bx + 2) * (by + 2)];
    volatile __shared__ float dv[(bx + 2) * (by + 2)];

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // position within global memory array
    const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

    // position within shared memory array
    const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

    // Load data to shared memory.
    // load tile being processed
    du[shMemPos] = du0[pos];
    dv[shMemPos] = dv0[pos];

    // load necessary neighbouring elements
    // We clamp out-of-range coordinates.
    // It is equivalent to mirroring
    // because we access data only one step away from borders.
    if (threadIdx.y == 0) {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        x = min(bsx + threadIdx.x, w - 1);
        // row just below the tile
        y         = max(bsy - 1, 0);
        gmPos     = y * s + x;
        smPos     = threadIdx.x + 1;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];

        // row above the tile
        y = min(bsy + by, h - 1);
        smPos += (by + 1) * (bx + 2);
        gmPos     = y * s + x;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];
    }
    else if (threadIdx.y == 1) {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        y = min(bsy + threadIdx.x, h - 1);
        // column to the left
        x     = max(bsx - 1, 0);
        smPos = bx + 2 + threadIdx.x * (bx + 2);
        gmPos = x + y * s;

        // check if we are within tile
        if (threadIdx.x < by) {
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
            // column to the right
            x     = min(bsx + bx, w - 1);
            gmPos = y * s + x;
            smPos += bx + 1;
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
        }
    }

    cg::sync(cta);

    if (ix >= w || iy >= h)
        return;

    // now all necessary data are loaded to shared memory
    int left, right, up, down;
    left  = shMemPos - 1;
    right = shMemPos + 1;
    up    = shMemPos + bx + 2;
    down  = shMemPos - bx - 2;

    float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
    float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

    float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) / (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

    du1[pos] = sumU - Ix[pos] * frac;
    dv1[pos] = sumV - Iy[pos] * frac;
}

__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out, cudaTextureObject_t texCoarse)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)
        return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * stride] = tex2D<float>(texCoarse, x, y) * scale;
}

__global__ void WarpingKernel(int                 width,
                              int                 height,
                              int                 stride,
                              const float        *u,
                              const float        *v,
                              float              *out,
                              cudaTextureObject_t texToWarp)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float x = ((float)ix + u[pos] + 0.5f) / (float)width;
    float y = ((float)iy + v[pos] + 0.5f) / (float)height;

    out[pos] = tex2D<float>(texToWarp, x, y);
}
