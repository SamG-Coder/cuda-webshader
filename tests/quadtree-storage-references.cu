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

class Bounding_box
{
    // Extreme points of the bounding box.
    float2 m_p_min;
    float2 m_p_max;

public:
    // Constructor. Create a unit box.
    __host__ __device__ Bounding_box()
    {
        m_p_min = make_float2(0.0f, 0.0f);
        m_p_max = make_float2(1.0f, 1.0f);
    }

    // Compute the center of the bounding-box.
    __host__ __device__ void compute_center(float2 &center) const
    {
        center.x = 0.5f * (m_p_min.x + m_p_max.x);
        center.y = 0.5f * (m_p_min.y + m_p_max.y);
    }

    // The points of the box.
    __host__ __device__ __forceinline__ const float2 &get_max() const { return m_p_max; }

    __host__ __device__ __forceinline__ const float2 &get_min() const { return m_p_min; }

    // Does a box contain a point.
    __host__ __device__ bool contains(const float2 &p) const
    {
        return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y && p.y < m_p_max.y;
    }

    // Define the bounding box.
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y)
    {
        m_p_min.x = min_x;
        m_p_min.y = min_y;
        m_p_max.x = max_x;
        m_p_max.y = max_y;
    }
};

////////////////////////////////////////////////////////////////////////////////
// A node of a quadree.
////////////////////////////////////////////////////////////////////////////////
class Quadtree_node
{
    // The identifier of the node.
    int m_id;
    // The bounding box of the tree.
    Bounding_box m_bounding_box;
    // The range of points.
    int m_begin, m_end;

public:
    // Constructor.
    __host__ __device__ Quadtree_node()
        : m_id(0)
        , m_begin(0)
        , m_end(0)
    {
    }

    // The ID of a node at its level.
    __host__ __device__ int id() const { return m_id; }

    // The ID of a node at its level.
    __host__ __device__ void set_id(int new_id) { m_id = new_id; }

    // The bounding box.
    __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const { return m_bounding_box; }

    // Set the bounding box.
    __host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y)
    {
        m_bounding_box.set(min_x, min_y, max_x, max_y);
    }

    // The number of points in the tree.
    __host__ __device__ __forceinline__ int num_points() const { return m_end - m_begin; }

    // The range of points in the tree.
    __host__ __device__ __forceinline__ int points_begin() const { return m_begin; }

    __host__ __device__ __forceinline__ int points_end() const { return m_end; }

    // Define the range for that node.
    __host__ __device__ __forceinline__ void set_range(int begin, int end)
    {
        m_begin = begin;
        m_end   = end;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Algorithm parameters.
////////////////////////////////////////////////////////////////////////////////

__global__ void write_nodes(Quadtree_node *nodes,float *out){
 unsigned i=threadIdx.x;
 Quadtree_node &node=nodes[i];
 i=0;
 const Bounding_box &box=node.bounding_box();
 const float2 &corner=box.get_max();
 node.set_range(threadIdx.x*2,threadIdx.x*2+7);
 node.set_bounding_box(0.f,0.f,float(threadIdx.x)+2.f,float(threadIdx.x)+3.f);
 out[threadIdx.x*3]=corner.x;
 out[threadIdx.x*3+1]=corner.y;
 out[threadIdx.x*3+2]=node.num_points();
}
__global__ void read_nodes(Quadtree_node *nodes,float *out){
 const Quadtree_node &node=nodes[threadIdx.x];
 out[threadIdx.x*3]=node.bounding_box().get_max().x;
 out[threadIdx.x*3+1]=node.bounding_box().get_max().y;
 out[threadIdx.x*3+2]=node.num_points();
}

__global__ void write_alias_nodes(Quadtree_node *nodes,float *out){
 unsigned offset=threadIdx.x;
 Quadtree_node *base=&nodes[offset];
 Quadtree_node *alias=base+0;
 offset=0;
 Quadtree_node &node=alias[0];
 node.set_range(threadIdx.x*2,threadIdx.x*2+7);
 node.set_bounding_box(0.f,0.f,float(threadIdx.x)+2.f,float(threadIdx.x)+3.f);
 out[threadIdx.x*3]=node.bounding_box().get_max().x;
 out[threadIdx.x*3+1]=node.bounding_box().get_max().y;
 out[threadIdx.x*3+2]=node.num_points();
}
