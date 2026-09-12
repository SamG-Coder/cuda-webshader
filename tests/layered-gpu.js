export async function checkLayered(runtime){
 if(runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
 const load=async name=>new Float32Array(await(await fetch(new URL('../reports/layered-'+name+'.bin',import.meta.url))).arrayBuffer()),input=await load('input'),expected=await load('native'),texture=runtime.createLayeredTexture2D(input,{width:512,height:512,layers:5,format:'r32float',addressMode:'repeat'}),output=runtime.createBuffer(expected.byteLength);
 let distinct;
 try{
 const source=await(await fetch(new URL('./layered-kernel.cuh',import.meta.url))).text(),kernel=await runtime.kernel(source,{entry:'transformKernel',workgroupSize:[8,8,1]});
 const batch=runtime.batch();for(let layer=0;layer<5;layer++)batch.dispatch(kernel.bind({g_odata:output,tex:texture},{width:512,height:512,layer}),[64,64,1]);batch.submit();const actual=await runtime.read(output);for(let i=0;i<expected.length;i++)if(actual[i]!==expected[i])throw Error('Native layered mismatch '+i);
 distinct=runtime.createLayeredTexture2D(Float32Array.from({length:80},(_,i)=>Math.floor(i/16)*1000+i%16),{width:4,height:4,layers:5,format:'r32float',addressMode:'repeat'});
 const probe=await runtime.kernel('__device__ float read(cudaTextureObject_t t,int layer){return tex2DLayered<float>(t,0.375f,0.625f,layer);}__global__ void k(float* out,cudaTextureObject_t tex){out[threadIdx.x]=read(tex,threadIdx.x);}',{workgroupSize:[5]});runtime.batch().dispatch(probe.bind({out:output,tex:distinct},{}),[1]).submit();const selected=await runtime.read(output,Float32Array,20);for(let i=0;i<5;i++)if(selected[i]!==i*1000+9)throw Error('Layer selection or helper propagation mismatch');
 return {passed:true,device:runtime.describe(),width:512,height:512,layers:5,values:expected.length,nativeValuesExact:true,distinctLayerSelection:true,helperTexturePropagation:true,softwareAdapterRequested:false};
 }finally{runtime.destroyTexture(texture);if(distinct)runtime.destroyTexture(distinct);runtime.destroyBuffer(output);}
}
