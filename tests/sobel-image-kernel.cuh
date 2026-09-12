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

__device__ unsigned char ComputeSobel(unsigned char ul, // upper left
                                      unsigned char um, // upper middle
                                      unsigned char ur, // upper right
                                      unsigned char ml, // middle left
                                      unsigned char mm, // middle (unused)
                                      unsigned char mr, // middle right
                                      unsigned char ll, // lower left
                                      unsigned char lm, // lower middle
                                      unsigned char lr, // lower right
                                      float         fScale)
{
    short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
    short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
    short Sum  = (short)(fScale * (abs((int)Horz) + abs((int)Vert)));

    if (Sum < 0) {
        return 0;
    }
    else if (Sum > 0xff) {
        return 0xff;
    }

    return (unsigned char)Sum;
}


typedef unsigned char Pixel;

__global__ void
SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch, int w, int h, float fscale, cudaTextureObject_t tex)
{
    unsigned char *pSobel = (unsigned char *)(((char *)pSobelOriginal) + blockIdx.x * Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        pSobel[i] = min(max((tex2D<unsigned char>(tex, (float)i, (float)blockIdx.x) * fscale), 0.f), 255.f);
    }
}

__global__ void SobelTex(Pixel *pSobelOriginal, unsigned int Pitch, int w, int h, float fScale, cudaTextureObject_t tex)
{
    unsigned char *pSobel = (unsigned char *)(((char *)pSobelOriginal) + blockIdx.x * Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        unsigned char pix00 = tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x - 1);
        unsigned char pix01 = tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x - 1);
        unsigned char pix02 = tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x - 1);
        unsigned char pix10 = tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 0);
        unsigned char pix11 = tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 0);
        unsigned char pix12 = tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 0);
        unsigned char pix20 = tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 1);
        unsigned char pix21 = tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 1);
        unsigned char pix22 = tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 1);
        pSobel[i]           = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22, fScale);
    }
}

