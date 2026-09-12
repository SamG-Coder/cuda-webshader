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


typedef float2 cData;

__global__ void addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch)
{
    int    tx = threadIdx.x;
    int    ty = threadIdx.y;
    cData *fj = (cData *)((char *)v + (ty + spy) * pitch) + tx + spx;

    cData vterm = *fj;
    tx -= r;
    ty -= r;
    float s = 1.f / (1.f + tx * tx * tx * tx + ty * ty * ty * ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}

__global__ void updateVelocity_k(cData *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch)
{
    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float vxterm, vyterm;
    cData nvterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy) {
                int fjr = fi * pdx + gtidx;
                vxterm  = vx[fjr];
                vyterm  = vy[fjr];

                // Normalize the result of the inverse FFT
                float scale = 1.f / (dx * dy);
                nvterm.x    = vxterm * scale;
                nvterm.y    = vyterm * scale;

                cData *fj = (cData *)((char *)v + fi * pitch) + gtidx;
                *fj       = nvterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}

__global__ void advectParticles_k(cData *part, cData *v, int dx, int dy, float dt, int lb, size_t pitch)
{
    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    cData pterm, vterm;

    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy) {
                int fj = fi * dx + gtidx;
                pterm  = part[fj];

                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm   = *((cData *)((char *)v + yvi * pitch) + xvi);

                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;
                pterm.x += 1.f;
                pterm.x = pterm.x - (int)pterm.x;
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;
                pterm.y += 1.f;
                pterm.y = pterm.y - (int)pterm.y;

                part[fj] = pterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}
