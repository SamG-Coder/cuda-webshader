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
__global__ void transformKernel(float *g_odata, int width, cudaTextureObject_t tex)
{
    // calculate this thread's data point
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 0.5f offset and division are necessary to access the original data points
    // in the texture (such that bilinear interpolation will not be activated).
    // For details, see also CUDA Programming Guide, Appendix D

    float u = ((x + 0.5f) / (float)width) * 2.f - 1.f;
    float v = ((y + 0.5f) / (float)width) * 2.f - 1.f;

    float cx, cy, cz;

    for (unsigned int face = 0; face < 6; face++) {
        // Layer 0 is positive X face
        if (face == 0) {
            cx = 1;
            cy = -v;
            cz = -u;
        }
        // Layer 1 is negative X face
        else if (face == 1) {
            cx = -1;
            cy = -v;
            cz = u;
        }
        // Layer 2 is positive Y face
        else if (face == 2) {
            cx = u;
            cy = 1;
            cz = v;
        }
        // Layer 3 is negative Y face
        else if (face == 3) {
            cx = u;
            cy = -1;
            cz = -v;
        }
        // Layer 4 is positive Z face
        else if (face == 4) {
            cx = u;
            cy = -v;
            cz = 1;
        }
        // Layer 4 is negative Z face
        else if (face == 5) {
            cx = -u;
            cy = -v;
            cz = -1;
        }

        // read from texture, do expected transformation and write to global memory
        g_odata[face * width * width + y * width + x] = -texCubemap<float>(tex, cx, cy, cz);
    }
}


// MIT display helper: arrange original face results in a 3 by 2 atlas.
__global__ void arrangeFaces(float* input, float* image, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width * 3 && y < width * 2) {
        int face = (y / width) * 3 + x / width;
        image[y * width * 3 + x] = input[face * width * width + (y % width) * width + x % width];
    }
}
