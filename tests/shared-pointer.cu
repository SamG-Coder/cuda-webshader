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
// calcNormal below is unchanged NVIDIA sample code. Test entry and helpers are MIT.
__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
    float3 edge0 = *v1 - *v0;
    float3 edge1 = *v2 - *v0;
    // note - it's faster to perform normalization in vertex shader rather than
    // here
    return cross(edge0, edge1);
}
__device__ void mutate(float3* a, float3* b) { a[0].x += 1.0f; b[0].y += a[0].x; }
__device__ void advance(float3* p) { p += 1; p[0].z += 2.0f; }
__device__ void forward(float3* p) { advance(p); }
__global__ void sharedPointers(float4* output) {
 __shared__ float3 vertices[96];
 uint t=threadIdx.x, base=t*3, g=blockIdx.x*32+t;
 vertices[base]=make_float3(float(g),0,0);
 vertices[base+1]=make_float3(float(g)+1.0f,0,0);
 vertices[base+2]=make_float3(float(g),1,0);
 __syncthreads();
 float3 n=calcNormal(vertices+base,&vertices[base+1],&vertices[base+2]);
 float3 *pointers[3];
 pointers[0]=&vertices[base];
 pointers[1]=&vertices[base+1];
 pointers[2]=&vertices[base+2];
 n=n+calcNormal(pointers[0],pointers[1],pointers[2]);
 pointers[1]=&vertices[base];
 mutate(pointers[0],pointers[1]);
 forward(vertices+base);
 output[g*3]=make_float4(n,0.0f);
 output[g*3+1]=make_float4(vertices[base],1.0f);
 output[g*3+2]=make_float4(vertices[base+1],1.0f);
}
