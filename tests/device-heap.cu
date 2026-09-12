// MIT device allocation regression. Pointer storage survives separate launches.
struct Allocation { float2* values; int count; };
__global__ void allocate_values(Allocation* records,int* status) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 records[i].count=int(i%32u)+1;
 status[i]=cudaMalloc((void**)&records[i].values,records[i].count*sizeof(float2));
}
__global__ void write_values(Allocation* records) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 for(int j=0;j<records[i].count;j++)records[i].values[j]=make_float2(float(i),float(j));
}
__global__ void read_values(Allocation* records,float2* output) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 for(int j=0;j<records[i].count;j++)output[i*32u+(unsigned int)j]=records[i].values[j];
}
__global__ void free_values(Allocation* records,int* status) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 status[i]=cudaFree(records[i].values);
 records[i].values=NULL;
}
