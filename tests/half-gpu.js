export async function checkHalf(runtime) {
  if(!runtime.device.features.has('shader-f16'))throw Error('This test requires shader-f16');
  // E4 values are generated numerically; all nonzero scaled products are normal binary16.
  const decode=code=>{const magnitude=code&127,e=magnitude>>3,m=magnitude&7;return (code&128?-1:1)*(e?(1+m/8)*2**(e-7):m*2**-9);};
  const pairs=[];for(let a=0;a<256;a++)if((a&127)!==127)for(let b=0;b<256;b++)if((b&127)<=81)pairs.push(decode(a),decode(b));
  const source=`__global__ void products(const float* input,float* output,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;__shared__ __half2 tile[64];__half2 v=__floats2half2_rn(0.0f,0.0f);if(i<count)v=__floats2half2_rn(input[2*i]*4.0f,input[2*i+1]*4.0f);tile[threadIdx.x]=v;__syncthreads();if(i<count){float2 a=__half22float2(tile[threadIdx.x]);__half2 b=__floats2half2_rn(a.y,a.x);float2 p=__half22float2(__hmul2(tile[threadIdx.x],b));output[2*i]=p.x;output[2*i+1]=p.y;}}`;
  const kernel=await runtime.kernel(source,{workgroupSize:[64,1,1]}),input=runtime.createBuffer(new Float32Array(pairs)),output=runtime.createBuffer(pairs.length*4),count=pairs.length/2;
  try {
    runtime.batch().dispatch(kernel.bind({input,output},{count}),[Math.ceil(count/64),1,1]).submit();
    const result=await runtime.read(output);
    for(let i=0;i<count;i++){const expected=pairs[2*i]*pairs[2*i+1]*16;if(result[2*i]!==expected||result[2*i+1]!==expected)throw Error('Half product mismatch at pair '+i);}
    return {passed:true,pairs:count,lanes:count*2};
  }finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
}
