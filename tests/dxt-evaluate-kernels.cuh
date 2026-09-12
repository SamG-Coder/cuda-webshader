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
#define USE_TABLES 1
__constant__ float3 kColorMetric = {1.0f,1.0f,1.0f};
inline __device__ float3 roundAndExpand(float3 v, ushort *w)
{
    v.x = rintf(__saturatef(v.x) * 31.0f);
    v.y = rintf(__saturatef(v.y) * 63.0f);
    v.z = rintf(__saturatef(v.z) * 31.0f);

    *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
    v.x *= 0.03227752766457f; // approximate integer bit expansion.
    v.y *= 0.01583151765563f;
    v.z *= 0.03227752766457f;
    return v;
}

__constant__ float     alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
__constant__ float     alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
__constant__ const int prods4[4]      = {0x090000, 0x000900, 0x040102, 0x010402};
__constant__ const int prods3[4]      = {0x040000, 0x000400, 0x040101, 0x010401};

#define USE_TABLES 1

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
static __device__ float
evalPermutation4(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
// Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++) {
        const uint bits = permutation >> (2 * i);

        alphax_sum += alphaTable4[bits & 3] * colors[i];
        akku += prods4[bits & 3];
    }

    float  alpha2_sum    = float(akku >> 16);
    float  beta2_sum     = float((akku >> 8) & 0xff);
    float  alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum     = (9.0f * color_sum) - alphax_sum;
#else
    float  alpha2_sum    = 0.0f;
    float  beta2_sum     = 0.0f;
    float  alphabeta_sum = 0.0f;
    float3 alphax_sum    = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++) {
        const uint bits = permutation >> (2 * i);

        float beta = (bits & 1);

        if (bits & 2) {
            beta = (1 + beta) * (1.0f / 3.0f);
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    // alpha2, beta2, alphabeta and factor could be precomputed for each
    // permutation, but it's faster to recompute them.
    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.111111111111f) * dot(e, kColorMetric);
}

static __device__ float
evalPermutation3(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
// Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++) {
        const uint bits = permutation >> (2 * i);

        alphax_sum += alphaTable3[bits & 3] * colors[i];
        akku += prods3[bits & 3];
    }

    float  alpha2_sum    = float(akku >> 16);
    float  beta2_sum     = float((akku >> 8) & 0xff);
    float  alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum     = (4.0f * color_sum) - alphax_sum;
#else
    float  alpha2_sum    = 0.0f;
    float  beta2_sum     = 0.0f;
    float  alphabeta_sum = 0.0f;
    float3 alphax_sum    = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++) {
        const uint bits = permutation >> (2 * i);

        float beta = (bits & 1);

        if (bits & 2) {
            beta = 0.5f;
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.25f) * dot(e, kColorMetric);
}


__global__ void dxtEvaluate(float4*input,uint*permutations,float4*errors,uint4*endpoints,unsigned n){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 __shared__ float3 colors[16];
 if(threadIdx.x<16){float4 c=input[threadIdx.x];colors[threadIdx.x]=make_float3(c.x,c.y,c.z);}
 __syncthreads();
 if(i<n){float3 sum=make_float3(0.0f,0.0f,0.0f);for(int j=0;j<16;j++)sum+=colors[j];ushort a,b,c,d;float e4=evalPermutation4(colors,permutations[i],&a,&b,sum);float e3=evalPermutation3(colors,permutations[i],&c,&d,sum);errors[i]=make_float4(e4,e3,0,0);endpoints[i]=make_uint4(a,b,c,d);}
}
