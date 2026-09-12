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

namespace cg=cooperative_groups;
typedef unsigned int uint;
#define SHARED_SIZE_LIMIT 1024U

// Map to single instructions on G8x / G9x / G100
#define UMUL(a, b)    __umul24((a), (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))

__device__ inline void Comparator(uint &keyA, uint &valA, uint &keyB, uint &valB, uint dir)
{
    uint t;

    if ((keyA > keyB) == dir) {
        t    = keyA;
        keyA = keyB;
        keyB = t;
        t    = valA;
        valA = valB;
        valB = t;
    }
}

__global__ void
oddEvenMergeSortShared(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint dir)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory storage for one or more small vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    // Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0]                       = d_SrcKey[0];
    s_val[threadIdx.x + 0]                       = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size <= arrayLength; size <<= 1) {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);

        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], dir);
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                Comparator(s_key[pos - stride], s_val[pos - stride], s_key[pos + 0], s_val[pos + 0], dir);
        }
    }

    cg::sync(cta);
    d_DstKey[0]                       = s_key[threadIdx.x + 0];
    d_DstVal[0]                       = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Odd-even merge sort iteration kernel
// for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
__global__ void oddEvenMergeGlobal(uint *d_DstKey,
                                   uint *d_DstVal,
                                   uint *d_SrcKey,
                                   uint *d_SrcVal,
                                   uint  arrayLength,
                                   uint  size,
                                   uint  stride,
                                   uint  dir)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;

    // Odd-even merge
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    if (stride < size / 2) {
        uint offset = global_comparatorI & ((size / 2) - 1);

        if (offset >= stride) {
            uint keyA = d_SrcKey[pos - stride];
            uint valA = d_SrcVal[pos - stride];
            uint keyB = d_SrcKey[pos + 0];
            uint valB = d_SrcVal[pos + 0];

            Comparator(keyA, valA, keyB, valB, dir);

            d_DstKey[pos - stride] = keyA;
            d_DstVal[pos - stride] = valA;
            d_DstKey[pos + 0]      = keyB;
            d_DstVal[pos + 0]      = valB;
        }
    }
    else {
        uint keyA = d_SrcKey[pos + 0];
        uint valA = d_SrcVal[pos + 0];
        uint keyB = d_SrcKey[pos + stride];
        uint valB = d_SrcVal[pos + stride];

        Comparator(keyA, valA, keyB, valB, dir);

        d_DstKey[pos + 0]      = keyA;
        d_DstVal[pos + 0]      = valA;
        d_DstKey[pos + stride] = keyB;
        d_DstVal[pos + stride] = valB;
    }
}

////////////////////////////////////////////////////////////////////////////////
// MIT host-input helper: original NVIDIA value input is its element index.
__global__ void initializeSortValues(uint* values,uint N){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)values[i]=i;}
// MIT display helper: original keys on the left, sorted keys on the right.
__global__ void previewSort(const uint* original,const uint* sorted,uint* image){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<2097152u){uint x=i%2048u,row=i/2048u;image[i]=x<1024u?original[row*1024u+x]:sorted[row*1024u+x-1024u];}}
