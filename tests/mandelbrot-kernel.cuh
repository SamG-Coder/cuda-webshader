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

// Double single functions based on DSFUN90 package:
// http://crd.lbl.gov/~dhbailey/mpdist/index.html

// The dimensions of the thread block
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define ABS(n) ((n) < 0 ? -(n) : (n))
 
                                                     
                                                  
 
                                                                             
                   
                                                  
  
                   
                          
           
 
                                                                             
                   
__device__ inline void dsfeq(float &a0, float &a1, float b)
{
    a0 = b;
    a1 = 0.0f;
}          
 
                                     
__device__ inline void dsadd(float &c0, float &c1, const float a0, const float a1, const float b0, const float b1)
{
    // Compute dsa + dsb using Knuth's trick.
    float t1 = a0 + b0;
    float e  = t1 - a0;
    float t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;

    // The result is t1 + t2, after normalization.
    c0 = e = t1 + t2;
    c1     = t2 - (e - t1);
}          
 
                                     
__device__ inline void dssub(float &c0, float &c1, const float a0, const float a1, const float b0, const float b1)
{
    // Compute dsa - dsb using Knuth's trick.
    float t1 = a0 - b0;
    float e  = t1 - a0;
    float t2 = ((-b0 - e) + (a0 - (t1 - e))) + a1 - b1;

    // The result is t1 + t2, after normalization.
    c0 = e = t1 + t2;
    c1     = t2 - (e - t1);
}          
 
#if 1
 
                                                                          
__device__ inline void dsmul(float &c0, float &c1, const float a0, const float a1, const float b0, const float b1)
{
    // This splits dsa(1) and dsb(1) into high-order and low-order words.
    float cona = a0 * 8193.0f;
    float conb = b0 * 8193.0f;
    float sa1  = cona - (cona - a0);
    float sb1  = conb - (conb - b0);
    float sa2  = a0 - sa1;
    float sb2  = b0 - sb1;

    // Multilply a0 * b0 using Dekker's method.
    float c11 = a0 * b0;
    float c21 = (((sa1 * sb1 - c11) + sa1 * sb2) + sa2 * sb1) + sa2 * sb2;

    // Compute a0 * b1 + a1 * b0 (only high-order word is needed).
    float c2 = a0 * b1 + a1 * b0;

    // Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
    float t1 = c11 + c2;
    float e  = t1 - c11;
    float t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + a1 * b1;

    // The result is t1 + t2, after normalization.
    c0 = e = t1 + t2;
    c1     = t2 - (e - t1);
}          
 
#else
 
                                                                
                                                                          
 
                                                                             
                                                          
    
 
                                                                          
__device__ inline void dsmul(float &c0, float &c1, const float a0, const float a1, const float b0, const float b1)
{
    // This splits dsa(1) and dsb(1) into high-order and low-order words.
    float cona = a0 * 8193.0f;
    float conb = b0 * 8193.0f;
    float sa1  = cona - (cona - a0);
    float sb1  = conb - (conb - b0);
    float sa2  = a0 - sa1;
    float sb2  = b0 - sb1;

    // Multilply a0 * b0 using Dekker's method.
    float c11 = __fmul_rn(a0, b0);
    float c21 = (((sa1 * sb1 - c11) + sa1 * sb2) + sa2 * sb1) + sa2 * sb2;

    // Compute a0 * b1 + a1 * b0 (only high-order word is needed).
    float c2 = __fmul_rn(a0, b1) + __fmul_rn(a1, b0);

    // Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
    float t1 = c11 + c2;
    float e  = t1 - c11;
    float t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + __fmul_rn(a1, b1);

    // The result is t1 + t2, after normalization.
    c0 = e = t1 + t2;
    c1     = t2 - (e - t1);
}          
 
#endif
 
                                                     
#if 1
                    
template <class T>
__device__ inline int
CalcMandelbrot(const T xPos, const T yPos, const T xJParam, const T yJParam, const int crunch, const bool isJulia)
{
    T   x, y, xx, yy;
    int i = crunch;

    T xC, yC;

    if (isJulia) {
        xC = xJParam;
        yC = yJParam;
        y  = yPos;
        x  = xPos;
        yy = y * y;
        xx = x * x;
    }
    else {
        xC = xPos;
        yC = yPos;
        y  = 0;
        x  = 0;
        yy = 0;
        xx = 0;
    }

    do {
        // Iteration 1
        if (xx + yy > T(4.0))
            return i - 1;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 2
        if (xx + yy > T(4.0))
            return i - 2;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 3
        if (xx + yy > T(4.0))
            return i - 3;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 4
        if (xx + yy > T(4.0))
            return i - 4;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 5
        if (xx + yy > T(4.0))
            return i - 5;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 6
        if (xx + yy > T(4.0))
            return i - 6;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 7
        if (xx + yy > T(4.0))
            return i - 7;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 8
        if (xx + yy > T(4.0))
            return i - 8;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 9
        if (xx + yy > T(4.0))
            return i - 9;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 10
        if (xx + yy > T(4.0))
            return i - 10;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 11
        if (xx + yy > T(4.0))
            return i - 11;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 12
        if (xx + yy > T(4.0))
            return i - 12;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 13
        if (xx + yy > T(4.0))
            return i - 13;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 14
        if (xx + yy > T(4.0))
            return i - 14;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 15
        if (xx + yy > T(4.0))
            return i - 15;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 16
        if (xx + yy > T(4.0))
            return i - 16;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 17
        if (xx + yy > T(4.0))
            return i - 17;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 18
        if (xx + yy > T(4.0))
            return i - 18;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 19
        if (xx + yy > T(4.0))
            return i - 19;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 20
        i -= 20;

        if ((i <= 0) || (xx + yy > T(4.0)))
            return i;

        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;
    } while (1);
}                   
 
#else
 
template <class T>
__device__ inline int
CalcMandelbrot(const T xPos, const T yPos, const T xJParam, const T yJParam, const int crunch, const isJulia)
{
    T x, y, xx, yy, xC, yC;

    if (isJulia) {
        xC = xJParam;
        yC = yJParam;
        y  = yPos;
        x  = xPos;
        yy = y * y;
        xx = x * x;
    }
    else {
        xC = xPos;
        yC = yPos;
        y  = 0;
        x  = 0;
        yy = 0;
        xx = 0;
    }

    int i = crunch;

    while (--i && (xx + yy < T(4.0))) {
        y  = x * y * T(2.0) + yC;
        x  = xx - yy + xC;
        yy = y * y;
        xx = x * x;
    }

    return i; // i > 0 ? crunch - i : 0;
}                   
 
#endif
 
                                                                       
__device__ inline int CalcMandelbrotDS(const float xPos0,
                                       const float xPos1,
                                       const float yPos0,
                                       const float yPos1,
                                       const float xJParam,
                                       const float yJParam,
                                       const int   crunch,
                                       const bool  isJulia)
{
    float xx0, xx1;
    float yy0, yy1;
    float sum0, sum1;
    int   i = crunch;

    float x0, x1, y0, y1;
    float xC0, xC1, yC0, yC1;

    if (isJulia) {
        xC0 = xJParam;
        xC1 = 0;
        yC0 = yJParam;
        yC1 = 0;
        y0  = yPos0; // y = yPos;
        y1  = yPos1;
        x0  = xPos0; // x = xPos;
        x1  = xPos1;
        dsmul(yy0, yy1, y0, y1, y0, y1); // yy = y * y;
        dsmul(xx0, xx1, x0, x1, x0, x1); // xx = x * x;
    }
    else {
        xC0 = xPos0;
        xC1 = xPos1;
        yC0 = yPos0;
        yC1 = yPos1;
        y0  = 0; // y = 0 ;
        y1  = 0;
        x0  = 0; // x = 0 ;
        x1  = 0;
        yy0 = 0; // yy = 0 ;
        yy1 = 0;
        xx0 = 0; // xx = 0 ;
        xx1 = 0;
    }

    dsadd(sum0, sum1, xx0, xx1, yy0, yy1); // sum = xx + yy;

    while (--i && (sum0 + sum1 < 4.0f)) {
        dsmul(y0, y1, x0, x1, y0, y1); // y = x * y * 2.0f + yC;  // yC is yPos for
                                       // Mandelbrot and it is yJParam for Julia
        dsadd(y0, y1, y0, y1, y0, y1);
        dsadd(y0, y1, y0, y1, yC0, yC1);

        dssub(x0, x1, xx0, xx1, yy0, yy1); //  x = xx - yy + xC;  // xC is xPos for
                                           //  Mandelbrot and it is xJParam for
                                           //  Julia
        dsadd(x0, x1, x0, x1, xC0, xC1);

        dsmul(yy0, yy1, y0, y1, y0, y1);       // yy = y * y;
        dsmul(xx0, xx1, x0, x1, x0, x1);       // xx = x * x;
        dsadd(sum0, sum1, xx0, xx1, yy0, yy1); // sum = xx + yy;
    }

    return i;
}                     
 
                                                      
__device__ inline int CheckColors(const uchar4 &color0, const uchar4 &color1)
{
    int x = color1.x - color0.x;
    int y = color1.y - color0.y;
    int z = color1.z - color0.z;
    return (ABS(x) > 10) || (ABS(y) > 10) || (ABS(z) > 10);
}                
 
                                                                            
          
                                  
                                                                                             

                                                                
   
                                                                      
                                                                      
            
                                                                     
                                                                    
                                                                        
                                                                          
                                                                           
                                                                  
                                                                      
                                                                    
   
                                                                        
                                                                     
                                                                      
                                                                     
                                                                         
                                                                       
                                                                      
                                                                       
                                                                
                                                                         
                                                                        
    
 
                   
 
                                 
                               
                         
 
                                           
 
template <class T>
__global__ void Mandelbrot0(uchar4      *dst,
                            const int    imageW,
                            const int    imageH,
                            const int    crunch,
                            const T      xOff,
                            const T      yOff,
                            const T      xJP,
                            const T      yJP,
                            const T      scale,
                            const uchar4 colors,
                            const int    frame,
                            const int    animationFrame,
                            const int    gridWidth,
                            const int    numBlocks,
                            const bool   isJ)
{
    // loop until all blocks completed
    for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x) {
        unsigned int blockX = blockIndex % gridWidth;
        unsigned int blockY = blockIndex / gridWidth;

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;

        if ((ix < imageW) && (iy < imageH)) {
            // Calculate the location
            const T xPos = (T)ix * scale + xOff;
            const T yPos = (T)iy * scale + yOff;

            // Calculate the Mandelbrot index for the current location
            int m = CalcMandelbrot<T>(xPos, yPos, xJP, yJP, crunch, isJ);
            //            int m = blockIdx.x;         // uncomment to see scheduling
            //            order
            m = m > 0 ? crunch - m : 0;

            // Convert the Mandelbrot index into a color
            uchar4 color;

            if (m) {
                m += animationFrame;
                color.x = m * colors.x;
                color.y = m * colors.y;
                color.z = m * colors.z;
            }
            else {
                color.x = 0;
                color.y = 0;
                color.z = 0;
            }

            // Output the pixel
            int pixel = imageW * iy + ix;

            if (frame == 0) {
                color.w    = 0;
                dst[pixel] = color;
            }
            else {
                int frame1   = frame + 1;
                int frame2   = frame1 / 2;
                dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
                dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
                dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
            }
        }
    }

}                
 
                                                                   
__global__ void MandelbrotDS0(uchar4      *dst,
                              const int    imageW,
                              const int    imageH,
                              const int    crunch,
                              const float  xOff0,
                              const float  xOff1,
                              const float  yOff0,
                              const float  yOff1,
                              const float  xJP,
                              const float  yJP,
                              const float  scale,
                              const uchar4 colors,
                              const int    frame,
                              const int    animationFrame,
                              const int    gridWidth,
                              const int    numBlocks,
                              const bool   isJ)
{
    // loop until all blocks completed
    for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x) {
        unsigned int blockX = blockIndex % gridWidth;
        unsigned int blockY = blockIndex / gridWidth;

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;

        if ((ix < imageW) && (iy < imageH)) {
            // Calculate the location
            float xPos0 = (float)ix * scale;
            float xPos1 = 0.0f;
            float yPos0 = (float)iy * scale;
            float yPos1 = 0.0f;
            dsadd(xPos0, xPos1, xPos0, xPos1, xOff0, xOff1);
            dsadd(yPos0, yPos1, yPos0, yPos1, yOff0, yOff1);

            // Calculate the Mandelbrot index for the current location
            int m = CalcMandelbrotDS(xPos0, xPos1, yPos0, yPos1, xJP, yJP, crunch, isJ);
            m     = m > 0 ? crunch - m : 0;

            // Convert the Mandelbrot index into a color
            uchar4 color;

            if (m) {
                m += animationFrame;
                color.x = m * colors.x;
                color.y = m * colors.y;
                color.z = m * colors.z;
            }
            else {
                color.x = 0;
                color.y = 0;
                color.z = 0;
            }

            // Output the pixel
            int pixel = imageW * iy + ix;

            if (frame == 0) {
                color.w    = 0;
                dst[pixel] = color;
            }
            else {
                int frame1   = frame + 1;
                int frame2   = frame1 / 2;
                dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
                dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
                dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
            }
        }
    }
}                  
 
                                                             
template <class T>
__global__ void Mandelbrot1(uchar4      *dst,
                            const int    imageW,
                            const int    imageH,
                            const int    crunch,
                            const T      xOff,
                            const T      yOff,
                            const T      xJP,
                            const T      yJP,
                            const T      scale,
                            const uchar4 colors,
                            const int    frame,
                            const int    animationFrame,
                            const int    gridWidth,
                            const int    numBlocks,
                            const bool   isJ)
{
    // loop until all blocks completed
    for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x) {
        unsigned int blockX = blockIndex % gridWidth;
        unsigned int blockY = blockIndex / gridWidth;

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;

        if ((ix < imageW) && (iy < imageH)) {
            // Get the current pixel color
            int    pixel      = imageW * iy + ix;
            uchar4 pixelColor = dst[pixel];
            int    count      = 0;

            // Search for pixels out of tolerance surrounding the current pixel
            if (ix > 0) {
                count += CheckColors(pixelColor, dst[pixel - 1]);
            }

            if (ix + 1 < imageW) {
                count += CheckColors(pixelColor, dst[pixel + 1]);
            }

            if (iy > 0) {
                count += CheckColors(pixelColor, dst[pixel - imageW]);
            }

            if (iy + 1 < imageH) {
                count += CheckColors(pixelColor, dst[pixel + imageW]);
            }

            if (count) {
                // Calculate the location
                const T xPos = (T)ix * scale + xOff;
                const T yPos = (T)iy * scale + yOff;

                // Calculate the Mandelbrot index for the current location
                int m = CalcMandelbrot(xPos, yPos, xJP, yJP, crunch, isJ);
                m     = m > 0 ? crunch - m : 0;

                // Convert the Mandelbrot index into a color
                uchar4 color;

                if (m) {
                    m += animationFrame;
                    color.x = m * colors.x;
                    color.y = m * colors.y;
                    color.z = m * colors.z;
                }
                else {
                    color.x = 0;
                    color.y = 0;
                    color.z = 0;
                }

                // Output the pixel
                int frame1   = frame + 1;
                int frame2   = frame1 / 2;
                dst[pixel].x = (pixelColor.x * frame + color.x + frame2) / frame1;
                dst[pixel].y = (pixelColor.y * frame + color.y + frame2) / frame1;
                dst[pixel].z = (pixelColor.z * frame + color.z + frame2) / frame1;
            }
        }
    }

}                
 
                                                                            
            
__global__ void MandelbrotDS1(uchar4      *dst,
                              const int    imageW,
                              const int    imageH,
                              const int    crunch,
                              const float  xOff0,
                              const float  xOff1,
                              const float  yOff0,
                              const float  yOff1,
                              const float  xJP,
                              const float  yJP,
                              const float  scale,
                              const uchar4 colors,
                              const int    frame,
                              const int    animationFrame,
                              const int    gridWidth,
                              const int    numBlocks,
                              const bool   isJ)
{
    // loop until all blocks completed
    for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x) {
        unsigned int blockX = blockIndex % gridWidth;
        unsigned int blockY = blockIndex / gridWidth;

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;

        if ((ix < imageW) && (iy < imageH)) {
            // Get the current pixel color
            int    pixel      = imageW * iy + ix;
            uchar4 pixelColor = dst[pixel];
            int    count      = 0;

            // Search for pixels out of tolerance surrounding the current pixel
            if (ix > 0) {
                count += CheckColors(pixelColor, dst[pixel - 1]);
            }

            if (ix + 1 < imageW) {
                count += CheckColors(pixelColor, dst[pixel + 1]);
            }

            if (iy > 0) {
                count += CheckColors(pixelColor, dst[pixel - imageW]);
            }

            if (iy + 1 < imageH) {
                count += CheckColors(pixelColor, dst[pixel + imageW]);
            }

            if (count) {
                // Calculate the location
                float xPos0 = (float)ix * scale;
                float xPos1 = 0.0f;
                float yPos0 = (float)iy * scale;
                float yPos1 = 0.0f;
                dsadd(xPos0, xPos1, xPos0, xPos1, xOff0, xOff1);
                dsadd(yPos0, yPos1, yPos0, yPos1, yOff0, yOff1);

                // Calculate the Mandelbrot index for the current location
                int m = CalcMandelbrotDS(xPos0, xPos1, yPos0, yPos1, xJP, yJP, crunch, isJ);
                m     = m > 0 ? crunch - m : 0;

                // Convert the Mandelbrot index into a color
                uchar4 color;

                if (m) {
                    m += animationFrame;
                    color.x = m * colors.x;
                    color.y = m * colors.y;
                    color.z = m * colors.z;
                }
                else {
                    color.x = 0;
                    color.y = 0;
                    color.z = 0;
                }

                // Output the pixel
                int frame1   = frame + 1;
                int frame2   = frame1 / 2;
                dst[pixel].x = (pixelColor.x * frame + color.x + frame2) / frame1;
                dst[pixel].y = (pixelColor.y * frame + color.y + frame2) / frame1;
                dst[pixel].z = (pixelColor.z * frame + color.z + frame2) / frame1;
            }
        }
    }

}
