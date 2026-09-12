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

// Implements interlace NV12 frames batch resize

typedef unsigned char uint8_t;
__global__ static void resizeNV12BatchKernel(cudaTextureObject_t texSrcLuma,
                                             cudaTextureObject_t texSrcChroma,
                                             uint8_t            *pDstNv12,
                                             int                 nSrcWidth,
                                             int                 nSrcHeight,
                                             int                 nDstPitch,
                                             int                 nDstWidth,
                                             int                 nDstHeight,
                                             int                 nBatchSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int px = x * 2, py = y * 2;

    if ((px + 1) >= nDstWidth || (py + 1) >= nDstHeight)
        return;

    float fxScale = 1.0f * nSrcWidth / nDstWidth;
    float fyScale = 1.0f * nSrcHeight / nDstHeight;

    uint8_t *p            = pDstNv12 + px + py * nDstPitch;
    int      hh           = nDstHeight * 3 / 2;
    int      nByte        = nDstPitch * hh;
    int      px_fxScale   = px * fxScale;
    int      px_fxScale_1 = (px + 1) * fxScale;
    int      py_fyScale   = py * fyScale;
    int      py_fyScale_1 = (py + 1) * fyScale;

    for (int i = blockIdx.z; i < nBatchSize; i += gridDim.z) {
        *(uchar2 *)p               = make_uchar2(tex2D<uint8_t>(texSrcLuma, px_fxScale, py_fyScale),
                                   tex2D<uint8_t>(texSrcLuma, px_fxScale_1, py_fyScale));
        *(uchar2 *)(p + nDstPitch) = make_uchar2(tex2D<uint8_t>(texSrcLuma, px_fxScale, py_fyScale_1),
                                                 tex2D<uint8_t>(texSrcLuma, px_fxScale_1, py_fyScale_1));
        *(uchar2 *)(p + (nDstHeight - y) * nDstPitch) =
            tex2D<uchar2>(texSrcChroma, x * fxScale, (hh * i + nDstHeight + y) * fyScale);
        p += nByte;
        py += hh;
    }
}

