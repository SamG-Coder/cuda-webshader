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

template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos, typename vec4<T>::Type *positions, int numTiles, cg::thread_block cta)
{
    typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

// This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++) {
            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}

template <typename T>
__global__ void integrateBodies(typename vec4<T>::Type *__restrict__ newPos,
                                typename vec4<T>::Type *__restrict__ oldPos,
                                typename vec4<T>::Type *vel,
                                unsigned int            deviceOffset,
                                unsigned int            deviceNumBodies,
                                float                   deltaTime,
                                float                   damping,
                                int                     numTiles)
{
    // Handle to thread block group
    cg::thread_block cta   = cg::this_thread_block();
    int              index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies) {
        return;
    }

    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position, oldPos, numTiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in
    // bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    typename vec4<T>::Type velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[deviceOffset + index] = position;
    vel[deviceOffset + index]    = velocity;
}

