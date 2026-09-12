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
__global__ void d_preintegrate(int                 layer,
                               float               steps,
                               cudaExtent          extent,
                               cudaTextureObject_t transferTex,
                               cudaTextureObject_t transferIntegrateTex,
                               cudaSurfaceObject_t transferLayerPreintSurf)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= extent.width || y >= extent.height) {
        return;
    }

    float sx = float(x) / float(extent.width);
    float sy = float(y) / float(extent.height);

    float smax = max(sx, sy);
    float smin = min(sx, sy);

    float4 iv;

    if (x != y) {
        // assumes square textures!
        float fracc = smax - smin;
        fracc       = 1.0 / (fracc * steps);

        float4 intmax = tex1D<float4>(transferIntegrateTex, smax);
        float4 intmin = tex1D<float4>(transferIntegrateTex, smin);
        iv.x          = (intmax.x - intmin.x) * fracc;
        iv.y          = (intmax.y - intmin.y) * fracc;
        iv.z          = (intmax.z - intmin.z) * fracc;
        // iv.w = (intmax.w - intmin.w)*fracc;
        iv.w = (1.0 - exp(-(intmax.w - intmin.w) * fracc));
    }
    else {
        float4 sample = tex1D<float4>(transferTex, smin);
        iv.x          = sample.x;
        iv.y          = sample.y;
        iv.z          = sample.z;
        // iv.w = sample.w;
        iv.w = (1.0 - exp(-sample.w));
    }

    iv.x = __saturatef(iv.x);
    iv.y = __saturatef(iv.y);
    iv.z = __saturatef(iv.z);
    iv.w = __saturatef(iv.w);

    // surface writes need byte offsets for x!
    surf2DLayeredwrite(iv, transferLayerPreintSurf, x * sizeof(float4), y, layer);
}

