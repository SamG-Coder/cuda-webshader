import {CompilerClient} from '../src/compiler/client.js';
export async function checkCloneWorker(runtime){
 const client=new CompilerClient();let buffer;
 try{const source=`__device__ void accumulate(float* p,float x){p[0]=${Array(900).fill('x').join('+')};}\n__global__ void deepHelper(float* output){unsigned i=threadIdx.x;accumulate(output+i,(float)(i+1));}`;
 const {artifact}=await client.compile(source,{entry:'deepHelper',workgroupSize:[16]});
 const kernel=await runtime.kernel(artifact);buffer=runtime.createBuffer(64);runtime.batch().dispatch(kernel.bind({output:buffer}),[1]).submit();const values=await runtime.read(buffer,Float32Array);for(let i=0;i<16;i++)if(values[i]!==900*(i+1))throw Error('Worker deep helper output mismatch');return {workerCompiled:true,terms:900,compared:16,exact:true};
 }finally{client.dispose();if(buffer)runtime.destroyBuffer(buffer);}
}
