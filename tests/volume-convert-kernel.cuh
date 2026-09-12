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
                                                                
   
                                                                      
                                                                      
            
                                                                     
                                                                    
                                                                        
                                                                          
                                                                           
                                                                  
                                                                      
                                                                    
   
                                                                        
                                                                     
                                                                      
                                                                     
                                                                         
                                                                       
                                                                      
                                                                       
                                                                
                                                                         
                                                                        
    
 
#ifndef _VOLUME_H_
                   
 
                          
 
typedef unsigned char VolumeType; 
 
           
  
 
                  
      
                                       
                                    
                                           
                                         
                                          
       
 
                                                                                
                                     
   
 
                                                                           
 
#ifdef __CUDACC__
 
                                                                               
    
 
template <typename T> struct VolumeTypeInfo
{
}; 
 
template <> struct VolumeTypeInfo<unsigned char>
{
    static const cudaTextureReadMode           readMode = cudaReadModeNormalizedFloat;
    static __inline__ __device__ unsigned char convert(float sampled)
    {
        return (unsigned char)(__saturatef(sampled) * 255.0);
    }
}; 
 
template <> struct VolumeTypeInfo<unsigned short>
{
    static const cudaTextureReadMode            readMode = cudaReadModeNormalizedFloat;
    static __inline__ __device__ unsigned short convert(float sampled)
    {
        return (unsigned short)(__saturatef(sampled) * 65535.0);
    }
}; 
 
template <> struct VolumeTypeInfo<float>
{
    static const cudaTextureReadMode   readMode = cudaReadModeElementType;
    static __inline__ __device__ float convert(float sampled) { return sampled; }
}; 
 
#endif
 
#endif

                                                                                
__global__ void conversionProbe(const float* input,uint* output,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=VolumeTypeInfo<VolumeType>::convert(input[i]);}

// MIT arithmetic validation entry, independent of NVIDIA methods.
__global__ void scaleProbe(const float* input,uint* output,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=(unsigned char)(input[i]*65535.0);}
