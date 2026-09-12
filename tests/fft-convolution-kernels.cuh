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

#define USE_TEXTURE 1
#define LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
typedef float2 fComplex;

__global__ void padKernel_kernel(float *d_Dst,
                                 float *d_Src,
                                 int    fftH,
                                 int    fftW,
                                 int    kernelH,
                                 int    kernelW,
                                 int    kernelY,
                                 int    kernelX
#if (USE_TEXTURE)
                                 ,
                                 cudaTextureObject_t texFloat
#endif
)
{
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y < kernelH && x < kernelW) {
        int ky = y - kernelY;

        if (ky < 0) {
            ky += fftH;
        }

        int kx = x - kernelX;

        if (kx < 0) {
            kx += fftW;
        }

        d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
    }
}

__global__ void padDataClampToBorder_kernel(float *d_Dst,
                                            float *d_Src,
                                            int    fftH,
                                            int    fftW,
                                            int    dataH,
                                            int    dataW,
                                            int    kernelH,
                                            int    kernelW,
                                            int    kernelY,
                                            int    kernelX
#if (USE_TEXTURE)
                                            ,
                                            cudaTextureObject_t texFloat
#endif
)
{
    const int y       = blockDim.y * blockIdx.y + threadIdx.y;
    const int x       = blockDim.x * blockIdx.x + threadIdx.x;
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;

    if (y < fftH && x < fftW) {
        int dy, dx;

        if (y < dataH) {
            dy = y;
        }

        if (x < dataW) {
            dx = x;
        }

        if (y >= dataH && y < borderH) {
            dy = dataH - 1;
        }

        if (x >= dataW && x < borderW) {
            dx = dataW - 1;
        }

        if (y >= borderH) {
            dy = 0;
        }

        if (x >= borderW) {
            dx = 0;
        }

        d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
    }
}

inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c)
{
    fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a          = t;
}

__global__ void modulateAndNormalize_kernel(fComplex *d_Dst, fComplex *d_Src, int dataSize, float c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= dataSize) {
        return;
    }

    fComplex a = d_Src[i];
    fComplex b = d_Dst[i];

    mulAndScale(a, b, c);

    d_Dst[i] = a;
}
