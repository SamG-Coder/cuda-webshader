#include <cooperative_groups.h>
#include <vector>
#include "flowCUDA.cu"
int main(){const int w=640,h=480,s=640;std::vector<float> a(w*h),b(w*h),u(w*h),v(w*h);for(int i=0;i<2;i++){char path[100];sprintf(path,"reports/optical-input-%d.bin",i);FILE*f=fopen(path,"rb");if(!f||fread(i?b.data():a.data(),4,w*h,f)!=w*h)return 2;fclose(f);}ComputeFlowCUDA(a.data(),b.data(),w,h,s,.2f,5,3,500,u.data(),v.data());for(int i=0;i<2;i++){char path[100];sprintf(path,"reports/optical-full-%c-native.bin",i?'v':'u');FILE*f=fopen(path,"wb");if(!f)return 2;fwrite(i?v.data():u.data(),4,w*h,f);fclose(f);}printf("Captured full original optical flow: 640x480, 5 levels, 3 warps, 500 Jacobi iterations.\n");}
