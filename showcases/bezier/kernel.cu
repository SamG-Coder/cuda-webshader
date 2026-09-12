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

                                                                
   
                                                                      
                                                                      
            
                                                                     
                                                                    
                                                                        
                                                                          
                                                                           
                                                                  
                                                                      
                                                                    
   
                                                                        
                                                                     
                                                                      
                                                                     
                                                                         
                                                                       
                                                                      
                                                                       
                                                                
                                                                         
                                                                        
    
 
                              
                         
                   
                    
 
                __device__ float2 operator+(float2 a, float2 b)
{
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
} 
 
                __device__ float2 operator-(float2 a, float2 b)
{
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
} 
 
                __device__ float2 operator*(float a, float2 b)
{
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
} 
 
                __device__ float length(float2 a) { return sqrtf(a.x * a.x + a.y * a.y); } 

#define MAX_TESSELLATION 32
struct BezierLine
{
    float2  CP[3];
    float2 *vertexPos;
    int     nVertices;
}; 
 
__global__ void computeBezierLinePositions(int lidx, BezierLine *bLines, int nTessPoints)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < nTessPoints) {
        float u   = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;

        float B3u[3];

        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;

        float2 position = {0, 0};

        for (int i = 0; i < 3; i++) {
            position = position + B3u[i] * bLines[lidx].CP[i];
        }

        bLines[lidx].vertexPos[idx] = position;
    }
} 
 
__global__ void computeBezierLinesCDP(BezierLine *bLines, int nLines)
{
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;

    if (lidx < nLines) {
        float curvature = length(bLines[lidx].CP[1] - 0.5f * (bLines[lidx].CP[0] + bLines[lidx].CP[2]))
                        / length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESSELLATION);

        if (bLines[lidx].vertexPos == NULL) {
            bLines[lidx].nVertices = nTessPoints;
            cudaMalloc((void **)&bLines[lidx].vertexPos, nTessPoints * sizeof(float2));
        }

        computeBezierLinePositions<<<ceilf((float)bLines[lidx].nVertices / 32.0f), 32>>>(
            lidx, bLines, bLines[lidx].nVertices);
    }
} 
 
__global__ void freeVertexMem(BezierLine *bLines, int nLines)
{
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;

    if (lidx < nLines)
        cudaFree(bLines[lidx].vertexPos);
} 
 
                                                           
  
                             
                               
                                                  
 
                                                                 
                                                                             
 
                                   
                                                                       
 
                                                                                        
                                                                          
          
               
                                                                                        
                               
                                                                                                                 
                                 
          
      
           
                                                            
 
                                                 
                                                                      
 
                                                                                            
                            
                                                                        
                       
              
 
                                                                                                 
          
      
                        
                        
                                                                                     
                                          
                            
      
 
                         
  

#define N_LINES   256
#define BLOCK_DIM 64
                                
  
                                                    
 
                          
 
                                        
                                  
 
                                      
                                                                   
                                                                   
          
 
                                                   
                                      
                                   
      
 
                                                               
                                   
                       
      
 
                          
                                                                                   
                                                                                                           
                                                                              
                                                                                                                    
                       
 
                                           
 
                                                                                                            
                                         
                       
 
                        
  
