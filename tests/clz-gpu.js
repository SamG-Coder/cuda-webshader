export async function checkClz(runtime){
 const data=new Uint32Array(4096);for(let i=0;i<64;i++)data[i]=i<32?2**i:(2**(i-32)-1)>>>0;
 let seed=42;for(let i=64;i<data.length;i++)data[i]=seed=(Math.imul(seed,1664525)+1013904223)>>>0;
 const kernel=await runtime.kernel('__global__ void clz(const unsigned* input,int* output){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;output[2*i]=__clz(input[i]);output[2*i+1]=__clz((int)input[i]);}',{workgroupSize:[64,1,1]});
 const input=runtime.createBuffer(data),output=runtime.createBuffer(data.length*8);
 try{runtime.batch().dispatch(kernel.bind({input,output}),[64,1,1]).submit();const result=await runtime.read(output,Int32Array);
 for(let i=0;i<data.length;i++)if(result[2*i]!==Math.clz32(data[i])||result[2*i+1]!==Math.clz32(data[i]))throw Error('clz mismatch '+i);
 return {passed:true,patterns:data.length,signedAndUnsigned:true};
 }finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
}
