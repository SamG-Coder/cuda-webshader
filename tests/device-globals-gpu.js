export async function checkDeviceGlobals(runtime){
 if(runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
 const source='__device__ int values[128];__device__ unsigned int counter[1];__device__ void update(){values[threadIdx.x]+=int(threadIdx.x)+1;atomicAdd(&counter[0],1u);}__global__ void k(){update();}';
 const kernel=await runtime.kernel(source,{workgroupSize:[128]}),values=runtime.createBuffer(new Int32Array(128)),counter=runtime.createBuffer(new Uint32Array(1)),small=runtime.createBuffer(4);
 try{
  let rejected=false;try{kernel.bind({values:small,counter});}catch{rejected=true;}if(!rejected)throw Error('Undersized device array accepted');
  const invocation=kernel.bind({values,counter},{});runtime.batch().dispatch(invocation,[1]).dispatch(invocation,[1]).submit();
  const actual=await runtime.read(values,Int32Array),count=await runtime.read(counter,Uint32Array);
  if(actual.some((v,i)=>v!==2*(i+1))||count[0]!==256)throw Error('Device global persistence or helper atomic mismatch');
  return {passed:true,device:runtime.describe(),elements:128,dispatches:2,helperAtomics:256,undersizedRejected:true,softwareAdapterRequested:false};
 }finally{for(const b of [values,counter,small])runtime.destroyBuffer(b);}
}
