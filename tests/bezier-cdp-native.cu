// MIT capture harness. All original NVIDIA device functions are included unchanged.
#include <vector>
#include <cmath>
#define main original_bezier_main
#include "../.local/nvidia-audit/cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu"
#undef main
__global__ void captureVertices(const BezierLine* lines,float2* out) {
    int i=blockIdx.x,j=threadIdx.x;
    if(j<lines[i].nVertices)out[i*MAX_TESSELLATION+j]=lines[i].vertexPos[j];
}

int main() {
    std::vector<BezierLine> lines(N_LINES);
    std::vector<float2> controls(N_LINES*3), vertices(N_LINES*MAX_TESSELLATION);
    std::vector<int> counts(N_LINES);
    float2 last={0,0};
    srand(1);
    for(int i=0;i<N_LINES;i++) {
        lines[i].CP[0]=last;
        for(int j=1;j<3;j++) {
            lines[i].CP[j].x=float(rand())/float(RAND_MAX);
            lines[i].CP[j].y=float(rand())/float(RAND_MAX);
        }
        last=lines[i].CP[2];lines[i].vertexPos=nullptr;lines[i].nVertices=0;
        for(int j=0;j<3;j++)controls[i*3+j]=lines[i].CP[j];
    }
    BezierLine* device;
    checkCudaErrors(cudaMalloc(&device,N_LINES*sizeof(BezierLine)));
    checkCudaErrors(cudaMemcpy(device,lines.data(),N_LINES*sizeof(BezierLine),cudaMemcpyHostToDevice));
    computeBezierLinesCDP<<<N_LINES/BLOCK_DIM,BLOCK_DIM>>>(device,N_LINES);
    checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(lines.data(),device,N_LINES*sizeof(BezierLine),cudaMemcpyDeviceToHost));
    float2* captured;
    checkCudaErrors(cudaMalloc(&captured,vertices.size()*sizeof(float2)));
    checkCudaErrors(cudaMemset(captured,0,vertices.size()*sizeof(float2)));
    captureVertices<<<N_LINES,MAX_TESSELLATION>>>(device,captured);
    checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(vertices.data(),captured,vertices.size()*sizeof(float2),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(captured));
    int total=0;double maxError=0;
    for(int i=0;i<N_LINES;i++) {
        auto& line=lines[i];counts[i]=line.nVertices;
        if(counts[i]<4||counts[i]>MAX_TESSELLATION||!line.vertexPos)return 2;
        total+=counts[i];
        for(int j=0;j<counts[i];j++) {
            double u=double(j)/(counts[i]-1),v=1-u;
            double x=v*v*line.CP[0].x+2*u*v*line.CP[1].x+u*u*line.CP[2].x;
            double y=v*v*line.CP[0].y+2*u*v*line.CP[1].y+u*u*line.CP[2].y;
            auto p=vertices[i*MAX_TESSELLATION+j];
            double e=fmax(fabs(p.x-x),fabs(p.y-y));if(!std::isfinite(e)||e>1e-6)return 3;maxError=fmax(maxError,e);
        }
    }
    freeVertexMem<<<N_LINES/BLOCK_DIM,BLOCK_DIM>>>(device,N_LINES);
    checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaFree(device));
    auto save=[](const char* name,const void* data,size_t bytes){FILE* f=fopen(name,"wb");if(!f)exit(4);if(fwrite(data,1,bytes,f)!=bytes)exit(5);fclose(f);};
    save("reports/bezier-cdp-controls.bin",controls.data(),controls.size()*sizeof(float2));
    save("reports/bezier-cdp-counts.bin",counts.data(),counts.size()*sizeof(int));
    save("reports/bezier-cdp-native.bin",vertices.data(),vertices.size()*sizeof(float2));
    printf("{\"passed\":true,\"curves\":%d,\"vertices\":%d,\"maxIndependentError\":%.12g,\"originalDeviceBodies\":true,\"childKernelLaunches\":true}\n",N_LINES,total,maxError);
}
