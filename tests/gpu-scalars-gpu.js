export async function checkGpuScalars(runtime){
 const source='__global__ void counter(unsigned int* counts,unsigned int value){counts[1]=value;} __global__ void fill(unsigned int* out,unsigned int n,unsigned int value){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=value;} __global__ void signedValue(int* out,int value){out[0]=value;}';
 const counter=await runtime.kernel(source,{entry:'counter',workgroupSize:[1]}),fill=await runtime.kernel(source,{entry:'fill',workgroupSize:[8]}),signed=await runtime.kernel(source,{entry:'signedValue',workgroupSize:[1]});
 const counts=runtime.createBuffer(8),out=runtime.createBuffer(32),first=runtime.createBuffer(32),signedOut=runtime.createBuffer(4);
 try{
  const invocation=fill.bind({out},{n:0,value:11},{scalarBuffers:{n:{resource:counts,offset:4}}}),before=runtime.stats.readbackBytes;
  const batch=runtime.batch();batch.dispatch(counter.bind({counts},{value:3}),[1]).dispatch(invocation,[1]).copy(out,first);
  batch.dispatch(counter.bind({counts},{value:5}),[1]).dispatch(invocation.setScalars({value:22}),[1]);batch.submit();
  if(runtime.stats.readbackBytes!==before)throw Error('GPU scalar was read through the CPU');
  const a=await runtime.read(first,Uint32Array),b=await runtime.read(out,Uint32Array);
  if(a.some((v,i)=>v!==(i<3?11:0))||b.some((v,i)=>v!==(i<5?22:0)))throw Error('GPU scalar dispatch snapshots differ');
  runtime.batch().dispatch(counter.bind({counts},{value:0xfffffffd}),[1]).dispatch(signed.bind({out:signedOut},{value:0},{scalarBuffers:{value:{resource:counts,offset:4}}}),[1]).submit();
  if((await runtime.read(signedOut,Int32Array))[0]!==-3)throw Error('Signed GPU scalar bits changed');
  for(const scalarBuffers of [{missing:{resource:counts}},{n:{resource:counts,offset:1}},{n:{resource:counts,offset:8}}]){
   let rejected=false;try{fill.bind({out},{n:0,value:0},{scalarBuffers});}catch{rejected=true;}if(!rejected)throw Error('Invalid GPU scalar accepted');
  }
  return {gpuProducedCounts:true,dispatchSnapshots:true,signedBits:true,noIntermediateReadback:true,invalidInputsRejected:true};
 }finally{for(const buffer of [counts,out,first,signedOut])runtime.destroyBuffer(buffer);}
}
