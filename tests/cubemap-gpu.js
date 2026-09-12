export async function checkCubemap(runtime){
 if(runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
 const load=async name=>new Float32Array(await(await fetch(new URL('../reports/cubemap-'+name+'.bin',import.meta.url))).arrayBuffer()),input=await load('input'),expected=await load('native'),directions=await load('directions'),edges=await load('edges-default');
 const texture=runtime.createCubemapTexture(input,{width:64,addressMode:'repeat'}),output=runtime.createBuffer(expected.byteLength),coords=runtime.createBuffer(directions),edgeOutput=runtime.createBuffer(edges.byteLength);
 try{
 const source=await(await fetch(new URL('./cubemap-kernel.cuh',import.meta.url))).text(),kernel=await runtime.kernel(source,{entry:'transformKernel',workgroupSize:[8,8,1]});
 runtime.batch().dispatch(kernel.bind({g_odata:output,tex:texture},{width:64}),[8,8,1]).submit();const actual=await runtime.read(output);let maxError=0;for(let i=0;i<expected.length;i++){maxError=Math.max(maxError,Math.abs(actual[i]-expected[i]));if(actual[i]!==expected[i])throw Error('Cubemap face mismatch '+i+': '+actual[i]+' vs '+expected[i]);}
 const probe=await runtime.kernel('__device__ float sample(cudaTextureObject_t tex,float4 d){return texCubemap<float>(tex,d.x,d.y,d.z);}__global__ void k(float* out,float4* dirs,cudaTextureObject_t tex){out[threadIdx.x]=sample(tex,dirs[threadIdx.x]);}',{workgroupSize:[12]});
 runtime.batch().dispatch(probe.bind({out:edgeOutput,dirs:coords,tex:texture},{}),[1]).submit();const edgeActual=await runtime.read(edgeOutput);for(let i=0;i<12;i++)if(edgeActual[i]!==edges[i])throw Error('Default cubemap edge mismatch '+i+': '+edgeActual[i]+' vs '+edges[i]);
 let rejected=false;try{runtime.createCubemapTexture(input,{width:64,seamless:true});}catch{rejected=true;}if(!rejected)throw Error('Unsupported seamless filtering accepted');
 return {passed:true,device:runtime.describe(),faces:6,width:64,values:expected.length,maxAbsoluteError:maxError,edgeProbes:12,edgeExact:true,helperTexturePropagation:true,seamlessExplicitlyRejected:true,softwareAdapterRequested:false};
 }finally{runtime.destroyTexture(texture);for(const b of [output,coords,edgeOutput])runtime.destroyBuffer(b);}
}
