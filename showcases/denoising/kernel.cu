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

typedef unsigned int TColor;

////////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////
#define KNN_WINDOW_RADIUS   3
#define NLM_WINDOW_RADIUS   3
#define NLM_BLOCK_RADIUS    3
#define KNN_WINDOW_AREA     ((2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1))
#define NLM_WINDOW_AREA     ((2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1))
#define INV_KNN_WINDOW_AREA (1.0f / (float)KNN_WINDOW_AREA)
#define INV_NLM_WINDOW_AREA (1.0f / (float)NLM_WINDOW_AREA)

#define KNN_WEIGHT_THRESHOLD 0.02f
#define KNN_LERP_THRESHOLD   0.79f
#define NLM_WEIGHT_THRESHOLD 0.10f
#define NLM_LERP_THRESHOLD   0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8



namespace cg = cooperative_groups;


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

__device__ float lerpf(float a, float b, float c) { return a + (b - a) * c; }

__device__ float vecLen(float4 a, float4 b)
{
    return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
}

__device__ TColor make_color(float r, float g, float b, float a)
{
    return ((int)(a * 255.0f) << 24) | ((int)(b * 255.0f) << 16) | ((int)(g * 255.0f) << 8) | ((int)(r * 255.0f) << 0);
}

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

__global__ void Copy(TColor *dst, int imageW, int imageH, cudaTextureObject_t texImage)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        float4 fresult        = tex2D<float4>(texImage, x, y);
        dst[imageW * iy + ix] = make_color(fresult.x, fresult.y, fresult.z, 0);
    }
}

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

__global__ void KNN(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Normalized counter for the weight threshold
        float fCount = 0;
        // Total sum of pixel weights
        float sumWeights = 0;
        // Result accumulator
        float3 clr = {0, 0, 0};
        // Center of the KNN window
        float4 clr00 = tex2D<float4>(texImage, x, y);

        // Cycle through KNN window, surrounding (x, y) texel
        for (float i = -KNN_WINDOW_RADIUS; i <= KNN_WINDOW_RADIUS; i++)
            for (float j = -KNN_WINDOW_RADIUS; j <= KNN_WINDOW_RADIUS; j++) {
                float4 clrIJ      = tex2D<float4>(texImage, x + j, y + i);
                float  distanceIJ = vecLen(clr00, clrIJ);

                // Derive final weight from color distance
                float weightIJ = __expf(-(distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA));

                // Accumulate (x + j, y + i) texel color with computed weight
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                // Sum of weights for color normalization to [0..1] range
                sumWeights += weightIJ;

                // Update weight counter, if KNN weight for current window texel
                // exceeds the weight threshold
                fCount += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
            }

        // Normalize result color by sum of weights
        sumWeights = 1.0f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        // Choose LERP quotient basing on how many texels
        // within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        // Write final result to global memory
        clr.x                 = lerpf(clr.x, clr00.x, lerpQ);
        clr.y                 = lerpf(clr.y, clr00.y, lerpQ);
        clr.z                 = lerpf(clr.z, clr00.z, lerpQ);
        dst[imageW * iy + ix] = make_color(clr.x, clr.y, clr.z, 0);
    };
}

__global__ void KNNdiag(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Normalized counter for the weight threshold
        float fCount = 0;
        // Center of the KNN window
        float4 clr00 = tex2D<float4>(texImage, x, y);

        // Cycle through KNN window, surrounding (x, y) texel
        for (float i = -KNN_WINDOW_RADIUS; i <= KNN_WINDOW_RADIUS; i++)
            for (float j = -KNN_WINDOW_RADIUS; j <= KNN_WINDOW_RADIUS; j++) {
                float4 clrIJ      = tex2D<float4>(texImage, x + j, y + i);
                float  distanceIJ = vecLen(clr00, clrIJ);

                // Derive final weight from color and geometric distance
                float weightIJ = __expf(-(distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA));

                // Update weight counter, if KNN weight for current window texel
                // exceeds the weight threshold
                fCount += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0.0f;
            }

        // Choose LERP quotient basing on how many texels
        // within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? 1.0f : 0;

        // Write final result to global memory
        dst[imageW * iy + ix] = make_color(lerpQ, 0, (1.0f - lerpQ), 0);
    };
}

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

__global__ void NLM(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Normalized counter for the NLM weight threshold
        float fCount = 0;
        // Total sum of pixel weights
        float sumWeights = 0;
        // Result accumulator
        float3 clr = {0, 0, 0};

        // Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS; j++) {
                // Find color distance from (x, y) to (x + j, y + i)
                float weightIJ = 0;

                for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
                    for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                        weightIJ += vecLen(tex2D<float4>(texImage, x + j + m, y + i + n),
                                           tex2D<float4>(texImage, x + m, y + n));

                // Derive final weight from color and geometric distance
                weightIJ = __expf(-(weightIJ * Noise + (i * i + j * j) * INV_NLM_WINDOW_AREA));

                // Accumulate (x + j, y + i) texel color with computed weight
                float4 clrIJ = tex2D<float4>(texImage, x + j, y + i);
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                // Sum of weights for color normalization to [0..1] range
                sumWeights += weightIJ;

                // Update weight counter, if NLM weight for current window texel
                // exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        // Normalize result color by sum of weights
        sumWeights = 1.0f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        // Choose LERP quotient basing on how many texels
        // within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        // Write final result to global memory
        float4 clr00          = tex2D<float4>(texImage, x, y);
        clr.x                 = lerpf(clr.x, clr00.x, lerpQ);
        clr.y                 = lerpf(clr.y, clr00.y, lerpQ);
        clr.z                 = lerpf(clr.z, clr00.z, lerpQ);
        dst[imageW * iy + ix] = make_color(clr.x, clr.y, clr.z, 0);
    }
}

__global__ void
NLMdiag(TColor *dst, unsigned int imageW, unsigned int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Normalized counter for the weight threshold
        float fCount = 0;

        // Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS; j++) {
                // Find color distance between (x, y) and (x + j, y + i)
                float weightIJ = 0;

                for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
                    for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                        weightIJ += vecLen(tex2D<float4>(texImage, x + j + m, y + i + n),
                                           tex2D<float4>(texImage, x + m, y + n));

                // Derive final weight from color and geometric distance
                weightIJ = __expf(-(weightIJ * Noise + (i * i + j * j) * INV_NLM_WINDOW_AREA));

                // Increase the weight threshold counter
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        // Choose LERP quotient basing on how many texels
        // within the NLM window exceeded the LERP threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? 1.0f : 0;

        // Write final result to global memory
        dst[imageW * iy + ix] = make_color(lerpQ, 0, (1.0f - lerpQ), 0);
    };
}

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

__global__ void NLM2(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // Weights cache
    __shared__ float fWeights[BLOCKDIM_X * BLOCKDIM_Y];

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x  = (float)ix + 0.5f;
    const float y  = (float)iy + 0.5f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 0.5f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Find color distance from current texel to the center of NLM window
        float weight = 0;

        for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                weight += vecLen(tex2D<float4>(texImage, cx + m, cy + n), tex2D<float4>(texImage, x + m, y + n));

        // Geometric distance from current texel to the center of NLM window
        float dist = (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS)
                   + (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        // Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        // Write the result to shared memory
        fWeights[threadIdx.y * BLOCKDIM_X + threadIdx.x] = weight;
        // Wait until all the weights are ready
        cg::sync(cta);

        // Normalized counter for the NLM weight threshold
        float fCount = 0;
        // Total sum of pixel weights
        float sumWeights = 0;
        // Result accumulator
        float3 clr = {0, 0, 0};

        int idx = 0;

        // Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++) {
                // Load precomputed weight
                float weightIJ = fWeights[idx++];

                // Accumulate (x + j, y + i) texel color with computed weight
                float4 clrIJ = tex2D<float4>(texImage, x + j, y + i);
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                // Sum of weights for color normalization to [0..1] range
                sumWeights += weightIJ;

                // Update weight counter, if NLM weight for current window texel
                // exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        // Normalize result color by sum of weights
        sumWeights = 1.0f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        // Choose LERP quotient basing on how many texels
        // within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        // Write final result to global memory
        float4 clr00          = tex2D<float4>(texImage, x, y);
        clr.x                 = lerpf(clr.x, clr00.x, lerpQ);
        clr.y                 = lerpf(clr.y, clr00.y, lerpQ);
        clr.z                 = lerpf(clr.z, clr00.z, lerpQ);
        dst[imageW * iy + ix] = make_color(clr.x, clr.y, clr.z, 0);
    }
}

__global__ void NLM2diag(TColor *dst, int imageW, int imageH, float Noise, float LerpC, cudaTextureObject_t texImage)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // Weights cache
    __shared__ float fWeights[BLOCKDIM_X * BLOCKDIM_Y];

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x  = (float)ix + 0.5f;
    const float y  = (float)iy + 0.5f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 0.5f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Find color distance from current texel to the center of NLM window
        float weight = 0;

        for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS + 1; n++)
            for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS + 1; m++)
                weight += vecLen(tex2D<float4>(texImage, cx + m, cy + n), tex2D<float4>(texImage, x + m, y + n));

        // Geometric distance from current texel to the center of NLM window
        float dist = (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS)
                   + (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        // Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        // Write the result to shared memory
        fWeights[threadIdx.y * BLOCKDIM_X + threadIdx.x] = weight;
        // Wait until all the weights are ready
        cg::sync(cta);

        // Normalized counter for the NLM weight threshold
        float fCount = 0;
        int   idx    = 0;

        // Cycle through NLM window, surrounding (x, y) texel
        for (float n = -NLM_WINDOW_RADIUS; n <= NLM_WINDOW_RADIUS + 1; n++)
            for (float m = -NLM_WINDOW_RADIUS; m <= NLM_WINDOW_RADIUS + 1; m++) {
                // Load precomputed weight
                float weightIJ = fWeights[idx++];

                // Update weight counter, if NLM weight for current window texel
                // exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        // Choose LERP quotient basing on how many texels
        // within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? 1.0f : 0.0f;

        // Write final result to global memory
        dst[imageW * iy + ix] = make_color(lerpQ, 0, (1.0f - lerpQ), 0);
    };
}
