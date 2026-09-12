// Project regression wrapper, MIT. Invokes NVIDIA's unchanged interaction helper.
__global__ void testBodyInteraction(float4* out,unsigned int n,unsigned int offset){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){float4 p=make_float4((i%8u)*0.125f,(i%3u)*0.25f,(i%5u)*0.125f,1.0f);
 float4 other=make_float4(2.0f,-1.0f,0.5f,1.5f);
 float3 acceleration=bodyBodyInteraction<float>(make_float3(0.0f,0.0f,0.0f),p,other);
 out[i+offset]=make_float4(acceleration.x,acceleration.y,acceleration.z,1.0f);}
}
