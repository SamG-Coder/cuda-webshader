// MIT ABI regression probe: adjacent bool bytes, native float3 alignment and padding.
struct Flags {bool a; bool b; float3 p; bool c; float w;};
__device__ float inspectFlags(const Flags* data) {
    return (data[0].a?100.f:0.f)+(data[0].b?10.f:0.f)+(data[0].c?1.f:0.f)
         +data[0].p.x+data[0].p.y+data[0].p.z+data[0].w;
}
__device__ void updateFlag(int& a,int& b) {a=4;b+=a;}
__global__ void nativeRecords(const Flags* data,float* output,int* flags) {
    output[0]=inspectFlags(data);
    output[1]=inspectFlags(data+1);
    int index=0;
    const Flags& value=data[index++];
    output[2]=value.w;
    output[3]=float(index);
    updateFlag(flags[1],flags[2]);
    updateFlag(flags[3],flags[3]);
    flags[4]=index;
}
