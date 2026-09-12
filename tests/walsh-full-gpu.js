export async function checkWalshFull(runtime){
 if(runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
 const source=await(await fetch(new URL('./walsh-full-kernels.cuh',import.meta.url))).text(),load=async name=>new Float32Array(await(await fetch(new URL('../reports/walsh-full-'+name+'.bin',import.meta.url))).arrayBuffer());
 const input=await load('input'),small=await load('kernel'),expected=await load('native'),N=8388608;
 if(input.length!==N||small.length!==128||expected.length!==N)throw Error('Wrong native workload');
 const padded=new Float32Array(N);padded.set(small);
 const data=runtime.createBuffer(input),filter=runtime.createBuffer(padded),aliases={d_Input:'d_Output'};
 try{
 const global=await runtime.kernel(source,{entry:'fwtBatch2Kernel',workgroupSize:[256],bufferAliases:aliases}),shared=await runtime.kernel(source,{entry:'fwtBatch1Kernel',workgroupSize:[512],sharedMemoryBytes:8192,bufferAliases:aliases}),modulate=await runtime.kernel(source,{entry:'modulateKernel',workgroupSize:[256]});
 const transform=buffer=>{const batch=runtime.batch();for(let stride=N/4;stride>=2048;stride/=4)batch.dispatch(global.bind({d_Output:buffer},{stride}),[8192,1,1]);batch.dispatch(shared.bind({d_Output:buffer},{log2N:11}),[4096,1,1]);batch.submit();};
 transform(data);transform(filter);runtime.batch().dispatch(modulate.bind({d_A:data,d_B:filter},{N}),[128,1,1]).submit();transform(data);
 const actual=await runtime.read(data);let delta=0,reference=0,maxError=0,bitMismatches=0;const aBits=new Uint32Array(actual.buffer),eBits=new Uint32Array(expected.buffer);
 for(let i=0;i<N;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite output '+i);const e=actual[i]-expected[i];delta+=e*e;reference+=expected[i]*expected[i];maxError=Math.max(maxError,Math.abs(e));if(aBits[i]!==eBits[i])bitMismatches++;}
 const l2=Math.sqrt(delta/reference);if(l2>=1e-6||maxError>1e-4)throw Error('Native mismatch L2 '+l2+' max '+maxError);
 return {passed:true,device:runtime.describe(),values:N,kernelValues:128,transforms:3,globalPassesPerTransform:6,sharedPassesPerTransform:1,l2,maxAbsoluteError:maxError,bitMismatches,softwareAdapterRequested:false};
 }finally{runtime.destroyBuffer(data);runtime.destroyBuffer(filter);}
}
