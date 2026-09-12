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

__device__ void d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++) {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++) {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    // main loop
    for (int x = (r + 1); x < w - r; x++) {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++) {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++) {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++) {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++) {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++) {
        t += id[(h - 1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}

// texture version
// texture fetches automatically clamp to edge of image
__global__ void d_boxfilter_x_tex(float *od, int w, int h, int r, cudaTextureObject_t tex)
{
    float        scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y     = blockIdx.x * blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int x = -r; x <= r; x++) {
        t += tex2D<float>(tex, x, y);
    }

    od[y * w] = t * scale;

    for (int x = 1; x < w; x++) {
        t += tex2D<float>(tex, x + r, y);
        t -= tex2D<float>(tex, x - r - 1, y);
        od[y * w + x] = t * scale;
    }
}

__global__ void d_boxfilter_y_tex(float *od, int w, int h, int r, cudaTextureObject_t tex)
{
    float        scale = 1.0f / (float)((r << 1) + 1);
    unsigned int x     = blockIdx.x * blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int y = -r; y <= r; y++) {
        t += tex2D<float>(tex, x, y);
    }

    od[x] = t * scale;

    for (int y = 1; y < h; y++) {
        t += tex2D<float>(tex, x, y + r);
        t -= tex2D<float>(tex, x, y - r - 1);
        od[y * w + x] = t * scale;
    }
}

// RGBA version
// reads from 32-bit unsigned int array holding 8-bit RGBA

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) | ((unsigned int)(rgba.z * 255.0f) << 16)
         | ((unsigned int)(rgba.y * 255.0f) << 8) | ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;         //  /255.0f;
    rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

// row pass using texture lookups
__global__ void d_boxfilter_rgba_x(unsigned int *od, int w, int h, int r, cudaTextureObject_t rgbaTex)
{
    float        scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y     = blockIdx.x * blockDim.x + threadIdx.x;

    // as long as address is always less than height, we do work
    if (y < h) {
        float4 t = make_float4(0.0f);

        for (int x = -r; x <= r; x++) {
            t += tex2D<float4>(rgbaTex, x, y);
        }

        od[y * w] = rgbaFloatToInt(t * scale);

        for (int x = 1; x < w; x++) {
            t += tex2D<float4>(rgbaTex, x + r, y);
            t -= tex2D<float4>(rgbaTex, x - r - 1, y);
            od[y * w + x] = rgbaFloatToInt(t * scale);
        }
    }
}

// column pass using coalesced global memory reads
__global__ void d_boxfilter_rgba_y(unsigned int *id, unsigned int *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    id             = &id[x];
    od             = &od[x];

    float scale = 1.0f / (float)((r << 1) + 1);

    float4 t;
    // do left edge
    t = rgbaIntToFloat(id[0]) * r;

    for (int y = 0; y < (r + 1); y++) {
        t += rgbaIntToFloat(id[y * w]);
    }

    od[0] = rgbaFloatToInt(t * scale);

    for (int y = 1; y < (r + 1); y++) {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[0]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++) {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // do right edge
    for (int y = h - r; y < h; y++) {
        t += rgbaIntToFloat(id[(h - 1) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }
}

