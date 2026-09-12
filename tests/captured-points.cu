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

class Points
{
    float *m_x;
    float *m_y;

public:
    // Constructor.
    __host__ __device__ Points()
        : m_x(NULL)
        , m_y(NULL)
    {
    }

    // Constructor.
    __host__ __device__ Points(float *x, float *y)
        : m_x(x)
        , m_y(y)
    {
    }

    // Get a point.
    __host__ __device__ __forceinline__ float2 get_point(int idx) const { return make_float2(m_x[idx], m_y[idx]); }

    // Set a point.
    __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p)
    {
        m_x[idx] = p.x;
        m_y[idx] = p.y;
    }

    // Set the pointers.
    __host__ __device__ __forceinline__ void set(float *x, float *y)
    {
        m_x = x;
        m_y = y;
    }
};

////////////////////////////////////////////////////////////////////////////////
// A 2D bounding box
////////////////////////////////////////////////////////////////////////////////

__global__ void bind_points(Points* pts,float* x0,float* y0,float* x1,float* y1){Points a(x0+2,y0+2);Points b(x1,y1);pts[0]=a;pts[1]=b;}
__global__ void update_points(Points* pts,int selector){int i=int(threadIdx.x);Points selected=pts[selector];float2 p=selected.get_point(i);p.x+=1.f;p.y-=2.f;selected.set_point(i,p);}
__global__ void read_points(Points* pts,float2* out,int selector){Points selected=pts[selector];out[threadIdx.x]=selected.get_point(int(threadIdx.x));}
