// SPDX-License-Identifier: MIT
#define SMEM(X, Y) sdata[(Y) * tilew + (X)]
#define ENABLED 0
__global__ void indexedTile(unsigned int* output) {
  __shared__ uchar4 sdata[4];
  int tilew = 2;
#ifndef DISABLED
  SMEM(1, 0) = make_uchar4(7, 11, 13, 255);
#endif
#ifdef ENABLED
  SMEM(0, 1) = SMEM(1, 0);
#endif
#ifdef SMEM
  uchar4 pixel = SMEM(0, 1);
  output[0] = pixel.x + pixel.y * 256u + pixel.z * 65536u + pixel.w * 16777216u;
#endif
}
