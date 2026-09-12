// MIT capture harness. NVIDIA implementation is included unchanged.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../.local/nvidia-audit/cpp/5_Domain_Specific/binomialOptions/binomialOptions_kernel.cu"
extern "C" void binomialOptionsCPU(real &, TOptionData);
template<class T> void save(const char* path,const T* data,size_t count) {
    FILE* f=fopen(path,"wb");
    if(!f||fwrite(data,sizeof(T),count,f)!=count){fprintf(stderr,"Capture write failed\n");exit(1);}
    fclose(f);
}
real randomData(real low,real high){real t=(real)rand()/(real)RAND_MAX;return ((real)1-t)*low+t*high;}
int main(){
    TOptionData input[MAX_OPTIONS];
    real output[MAX_OPTIONS],cpu[MAX_OPTIONS];
    __TOptionData prepared[MAX_OPTIONS];
    srand(123);
    for(int i=0;i<MAX_OPTIONS;i++){
        input[i].S=randomData(5.0f,30.0f);input[i].X=randomData(1.0f,100.0f);
        input[i].T=randomData(0.25f,10.0f);input[i].R=0.06f;input[i].V=0.10f;
    }
    binomialOptionsGPU(output,input,MAX_OPTIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyFromSymbol(prepared,d_OptionData,sizeof(prepared)));
    double delta=0,reference=0,maxError=0;
    for(int i=0;i<MAX_OPTIONS;i++){
        binomialOptionsCPU(cpu[i],input[i]);
        if(!std::isfinite(output[i])||output[i]<0){fprintf(stderr,"Invalid result\n");return 1;}
        const double error=fabs(double(output[i])-cpu[i]);
        delta+=error;reference+=fabs(cpu[i]);if(error>maxError)maxError=error;
    }
    const double l1=delta/reference;
    if(l1>5e-4){fprintf(stderr,"Original CPU tolerance failed: %.9g\n",l1);return 1;}
    save("reports/binomial-options-input.bin",input,MAX_OPTIONS);
    save("reports/binomial-options-prepared.bin",prepared,MAX_OPTIONS);
    save("reports/binomial-options-native.bin",output,MAX_OPTIONS);
    save("reports/binomial-options-cpu.bin",cpu,MAX_OPTIONS);
    printf("options=%d steps=%d native_vs_original_cpu_l1=%.9g max_absolute_error=%.9g passed\n",MAX_OPTIONS,NUM_STEPS,l1,maxError);
}
