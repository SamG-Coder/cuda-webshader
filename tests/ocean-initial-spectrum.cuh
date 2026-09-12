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
// Original host spectrum generation and parameters, isolated for native validation.
const unsigned meshSize=256,spectrumW=260,spectrumH=257;
const float g=9.81f,A=1e-7f,patchSize=100;
float windSpeed=100.f,windDir=CUDART_PI_F/3.f,dirDepend=.07f;
float urand() { return rand() / (float)RAND_MAX; }

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss()
{
    float u1 = urand();
    float u2 = urand();

    if (u1 < 1e-6f) {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
}

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend)
{
    float k_squared = Kx * Kx + Ky * Ky;

    if (k_squared == 0.0f) {
        return 0.0f;
    }

    // largest possible wave from constant wind of velocity v
    float L = V * V / g;

    float k_x     = Kx / sqrtf(k_squared);
    float k_y     = Ky / sqrtf(k_squared);
    float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

    float phillips = A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

    // filter out waves moving opposite to wind
    if (w_dot_k < 0.0f) {
        phillips *= dir_depend;
    }

    // damp out waves with very small length w << l
    // float w = L / 10000;
    // phillips *= expf(-k_squared * w * w);

    return phillips;
}

// Generate base heightfield in frequency space
void generate_h0(float2 *h0)
{
    for (unsigned int y = 0; y <= meshSize; y++) {
        for (unsigned int x = 0; x <= meshSize; x++) {
            float kx = (-(int)meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            float ky = (-(int)meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

            float P = sqrtf(phillips(kx, ky, windDir, windSpeed, A, dirDepend));

            if (kx == 0.0f && ky == 0.0f) {
                P = 0.0f;
            }

            // float Er = urand()*2.0f-1.0f;
            // float Ei = urand()*2.0f-1.0f;
            float Er = gauss();
            float Ei = gauss();

            float h0_re = Er * P * CUDART_SQRT_HALF_F;
            float h0_im = Ei * P * CUDART_SQRT_HALF_F;

            int i   = y * spectrumW + x;
            h0[i].x = h0_re;
            h0[i].y = h0_im;
        }
    }
}

