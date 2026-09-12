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

                                                                
   
                                                                      
                                                                      
            
                                                                     
                                                                    
                                                                        
                                                                          
                                                                           
                                                                  
                                                                      
                                                                    
   
                                                                        
                                                                     
                                                                      
                                                                     
                                                                         
                                                                       
                                                                      
                                                                       
                                                                
                                                                         
                                                                        
    
 
#ifndef _VOLUMEFILTER_KERNEL_H_
                                

#define VOLUMEFILTER_MAXWEIGHTS 125

                          
 
                    
 
           
  
                                                  
                                                    
                                                    
                                                       
                                                       
                                                    
                                                              
   
 
#endif

                                                                
   
                                                                      
                                                                      
            
                                                                     
                                                                    
                                                                        
                                                                          
                                                                           
                                                                  
                                                                      
                                                                    
   
                                                                        
                                                                     
                                                                      
                                                                     
                                                                         
                                                                       
                                                                      
                                                                       
                                                                
                                                                         
                                                                        
    
 
#ifndef _VOLUMEFILTER_KERNEL_CU_
                                 
 
                         
                         
 
                          
 
                             
                              
                               
 
__constant__ float4 c_filterData[VOLUMEFILTER_MAXWEIGHTS]; 
 
__global__ void d_filter_surface3d(int                 filterSize,
                                   float               filter_offset,
                                   cudaExtent          volumeSize,
                                   cudaTextureObject_t volumeTexIn,
                                   cudaSurfaceObject_t volumeTexOut)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth) {
        return;
    }

    float  filtered  = 0;
    float4 basecoord = make_float4(x, y, z, 0);

    for (int i = 0; i < filterSize; i++) {
        float4 coord = basecoord + c_filterData[i];
        filtered += tex3D<float>(volumeTexIn, coord.x, coord.y, coord.z) * c_filterData[i].w;
    }

    filtered += filter_offset;

    VolumeType output = VolumeTypeInfo<VolumeType>::convert(filtered);

    // surface writes need byte offsets for x!
    surf3Dwrite(output, volumeTexOut, x * sizeof(VolumeType), y, z);
} 
 
                                               
  
                                                       
                          
                                                                                 
                                                 
      
 
                                           
  
 
                                                         
                                                           
                                                           
                                                              
                                                              
                                                           
                                                                    
  
                           
                                     
                                                 
                                         
                                                                                                                       
 
                   
                                                                                             
 
                                           
                                                     
                                                                                        
 
                                                  
 
                         
                           
                        
 
                      
                               
          
      
 
                  
  
#endif
