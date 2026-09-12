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
#define POWER_OF_TWO 1
typedef float2 fComplex;
#define LOAD_FCOMPLEX(i) tex1Dfetch<fComplex>(texComplex, i)
#define LOAD_FCOMPLEX_A(i) tex1Dfetch<fComplex>(texComplexA, i)
#define LOAD_FCOMPLEX_B(i) tex1Dfetch<fComplex>(texComplexB, i)

inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c)
{
    fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a          = t;
}

inline __device__ void spPostprocessC2C(fComplex &D1, fComplex &D2, const fComplex &twiddle)
{
    float A1 = 0.5f * (D1.x + D2.x);
    float B1 = 0.5f * (D1.y - D2.y);
    float A2 = 0.5f * (D1.y + D2.y);
    float B2 = 0.5f * (D1.x - D2.x);

    D1.x = A1 + (A2 * twiddle.x + B2 * twiddle.y);
    D1.y = (A2 * twiddle.y - B2 * twiddle.x) + B1;
    D2.x = A1 - (A2 * twiddle.x + B2 * twiddle.y);
    D2.y = (A2 * twiddle.y - B2 * twiddle.x) - B1;
}

inline __device__ void spPreprocessC2C(fComplex &D1, fComplex &D2, const fComplex &twiddle)
{
    float A1 = /* 0.5f * */ (D1.x + D2.x);
    float B1 = /* 0.5f * */ (D1.y - D2.y);
    float A2 = /* 0.5f * */ (D1.y + D2.y);
    float B2 = /* 0.5f * */ (D1.x - D2.x);

    D1.x = A1 - (A2 * twiddle.x - B2 * twiddle.y);
    D1.y = (B2 * twiddle.x + A2 * twiddle.y) + B1;
    D2.x = A1 + (A2 * twiddle.x - B2 * twiddle.y);
    D2.y = (B2 * twiddle.x + A2 * twiddle.y) - B1;
}

inline __device__ void getTwiddle(fComplex &twiddle, float phase) { __sincosf(phase, &twiddle.y, &twiddle.x); }

inline __device__ uint mod(uint a, uint DA)
{
    //(DA - a) % DA, assuming a <= DA
    return a ? (DA - a) : a;
}

inline __device__ void udivmod(uint &dividend, uint divisor, uint &rem)
{
#if (!POWER_OF_TWO)
    rem = dividend % divisor;
    dividend /= divisor;
#else
    rem = dividend & (divisor - 1);
    dividend >>= (__ffs(divisor) - 1);
#endif
}

__global__ void spPostprocess2D_kernel(fComplex *d_Dst,
                                       fComplex *d_Src,
                                       uint      DY,
                                       uint      DX,
                                       uint      threadCount,
                                       uint      padding,
                                       float     phaseBase
#if (USE_TEXTURE)
                                       ,
                                       cudaTextureObject_t texComplex
#endif
)
{
    const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= threadCount) {
        return;
    }

    uint x, y, i = threadId;
    udivmod(i, DX / 2, x);
    udivmod(i, DY, y);

    // Avoid overwrites in columns DX / 2 by different threads
    if ((x == 0) && (y > DY / 2)) {
        return;
    }

    const uint srcOffset = i * DY * DX;
    const uint dstOffset = i * DY * (DX + padding);

    // Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
    {
        const uint loadPos1  = srcOffset + y * DX + x;
        const uint loadPos2  = srcOffset + mod(y, DY) * DX + mod(x, DX);
        const uint storePos1 = dstOffset + y * (DX + padding) + x;
        const uint storePos2 = dstOffset + mod(y, DY) * (DX + padding) + (DX - x);

        fComplex D1 = LOAD_FCOMPLEX(loadPos1);
        fComplex D2 = LOAD_FCOMPLEX(loadPos2);

        fComplex twiddle;
        getTwiddle(twiddle, phaseBase * (float)x);
        spPostprocessC2C(D1, D2, twiddle);

        d_Dst[storePos1] = D1;
        d_Dst[storePos2] = D2;
    }

    // Process x = DX / 2
    if (x == 0) {
        const uint loadPos1  = srcOffset + y * DX + DX / 2;
        const uint loadPos2  = srcOffset + mod(y, DY) * DX + DX / 2;
        const uint storePos1 = dstOffset + y * (DX + padding) + DX / 2;
        const uint storePos2 = dstOffset + mod(y, DY) * (DX + padding) + DX / 2;

        fComplex D1 = LOAD_FCOMPLEX(loadPos1);
        fComplex D2 = LOAD_FCOMPLEX(loadPos2);

        // twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(dir * j * PI / 2)
        fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
        spPostprocessC2C(D1, D2, twiddle);

        d_Dst[storePos1] = D1;
        d_Dst[storePos2] = D2;
    }
}

__global__ void spPreprocess2D_kernel(fComplex *d_Dst,
                                      fComplex *d_Src,
                                      uint      DY,
                                      uint      DX,
                                      uint      threadCount,
                                      uint      padding,
                                      float     phaseBase
#if (USE_TEXTURE)
                                      ,
                                      cudaTextureObject_t texComplex
#endif
)
{
    const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= threadCount) {
        return;
    }

    uint x, y, i = threadId;
    udivmod(i, DX / 2, x);
    udivmod(i, DY, y);

    // Avoid overwrites in columns 0 and DX / 2 by different threads (lower and
    // upper halves)
    if ((x == 0) && (y > DY / 2)) {
        return;
    }

    const uint srcOffset = i * DY * (DX + padding);
    const uint dstOffset = i * DY * DX;

    // Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
    {
        const uint loadPos1  = srcOffset + y * (DX + padding) + x;
        const uint loadPos2  = srcOffset + mod(y, DY) * (DX + padding) + (DX - x);
        const uint storePos1 = dstOffset + y * DX + x;
        const uint storePos2 = dstOffset + mod(y, DY) * DX + mod(x, DX);

        fComplex D1 = LOAD_FCOMPLEX(loadPos1);
        fComplex D2 = LOAD_FCOMPLEX(loadPos2);

        fComplex twiddle;
        getTwiddle(twiddle, phaseBase * (float)x);
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[storePos1] = D1;
        d_Dst[storePos2] = D2;
    }

    // Process x = DX / 2
    if (x == 0) {
        const uint loadPos1  = srcOffset + y * (DX + padding) + DX / 2;
        const uint loadPos2  = srcOffset + mod(y, DY) * (DX + padding) + DX / 2;
        const uint storePos1 = dstOffset + y * DX + DX / 2;
        const uint storePos2 = dstOffset + mod(y, DY) * DX + DX / 2;

        fComplex D1 = LOAD_FCOMPLEX(loadPos1);
        fComplex D2 = LOAD_FCOMPLEX(loadPos2);

        // twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(-dir * j * PI / 2)
        fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[storePos1] = D1;
        d_Dst[storePos2] = D2;
    }
}

__global__ void spProcess2D_kernel(fComplex *d_Dst,
                                   fComplex *d_SrcA,
                                   fComplex *d_SrcB,
                                   uint      DY,
                                   uint      DX,
                                   uint      threadCount,
                                   float     phaseBase,
                                   float     c
#if (USE_TEXTURE)
                                   ,
                                   cudaTextureObject_t texComplexA,
                                   cudaTextureObject_t texComplexB
#endif
)
{
    const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= threadCount) {
        return;
    }

    uint x, y, i = threadId;
    udivmod(i, DX, x);
    udivmod(i, DY / 2, y);

    const uint offset = i * DY * DX;

    // Avoid overwrites in rows 0 and DY / 2 by different threads (left and right
    // halves) Otherwise correctness for in-place transformations is affected
    if ((y == 0) && (x > DX / 2)) {
        return;
    }

    fComplex twiddle;

    // Process y = [0 .. DY / 2 - 1] U [DY - (DY / 2) + 1 .. DY - 1]
    {
        const uint pos1 = offset + y * DX + x;
        const uint pos2 = offset + mod(y, DY) * DX + mod(x, DX);

        fComplex D1 = LOAD_FCOMPLEX_A(pos1);
        fComplex D2 = LOAD_FCOMPLEX_A(pos2);
        fComplex K1 = LOAD_FCOMPLEX_B(pos1);
        fComplex K2 = LOAD_FCOMPLEX_B(pos2);
        getTwiddle(twiddle, phaseBase * (float)x);

        spPostprocessC2C(D1, D2, twiddle);
        spPostprocessC2C(K1, K2, twiddle);
        mulAndScale(D1, K1, c);
        mulAndScale(D2, K2, c);
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[pos1] = D1;
        d_Dst[pos2] = D2;
    }

    if (y == 0) {
        const uint pos1 = offset + (DY / 2) * DX + x;
        const uint pos2 = offset + (DY / 2) * DX + mod(x, DX);

        fComplex D1 = LOAD_FCOMPLEX_A(pos1);
        fComplex D2 = LOAD_FCOMPLEX_A(pos2);
        fComplex K1 = LOAD_FCOMPLEX_B(pos1);
        fComplex K2 = LOAD_FCOMPLEX_B(pos2);

        spPostprocessC2C(D1, D2, twiddle);
        spPostprocessC2C(K1, K2, twiddle);
        mulAndScale(D1, K1, c);
        mulAndScale(D2, K2, c);
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[pos1] = D1;
        d_Dst[pos2] = D2;
    }
}
