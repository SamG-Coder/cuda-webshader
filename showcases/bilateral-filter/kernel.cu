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

__constant__ float cGaussian[64];                                  
 
                                   
                                  
 
                                        
                                                   
              
 
   
                                       
 
                                                                      
                                                                         
                                                                       
                                                                        
                   
 
                                                                    
                                                                        
                                                                           
                                                                        
                                                       
 
                                                                            
                                                                        
                                
 
                                                                      
 
               
                                                 
                                           
                          
                     
                      
                       
   
 
                                                           
__device__ float euclideanLen(float4 a, float4 b, float d)
{
    float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
} 
 
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x)); // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8)
         | uint(rgba.x * 255.0f);
} 
 
__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;         //  /255.0f;
    rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
} 
 
                                                   
__global__ void d_bilateral_filter(uint *od, int w, int h, float e_d, int r, cudaTextureObject_t rgbaTex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) {
        return;
    }

    float  sum = 0.0f;
    float  factor;
    float4 t      = {0.f, 0.f, 0.f, 0.f};
    float4 center = tex2D<float4>(rgbaTex, x, y);

    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            float4 curPix = tex2D<float4>(rgbaTex, x + j, y + i);
            factor        = cGaussian[i + r] * cGaussian[j + r] * // domain factor
                     euclideanLen(curPix, center, e_d);           // range factor

            t += factor * curPix;
            sum += factor;
        }
    }

    od[y * w + x] = rgbaFloatToInt(t / sum);
}
