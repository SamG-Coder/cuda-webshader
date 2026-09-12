export async function checkPathtracerMaterials(runtime){
 const source=(await Promise.all(['class','kernels'].map(async n=>await(await fetch('/tests/pathtracer-material-'+n+'.cuh')).text()))).join('\n');
 let overallMax=0;const modes=[];
 for(const objectHeap of ['invocation','persistent']){
 const kernel=await runtime.kernel(source,{entry:'check_material',libraries:['curand-xorwow'],workgroupSize:[64,1,1],objectHeap}),arena=objectHeap==='persistent'?runtime.createObjectArena():null;
 const out=runtime.createBuffer(512*10*4),bits=runtime.createBuffer(512*4);
 try{
 runtime.batch().dispatch(kernel.bind({out,bits},{},arena?{objectArena:arena}:{}),[8,1,1]).submit();
 const actual=await runtime.read(out),actualBits=await runtime.read(bits,Uint32Array),expected=new Float32Array(await(await fetch('/reports/pathtracer-material-native.bin')).arrayBuffer()),expectedBits=new Uint32Array(await(await fetch('/reports/pathtracer-material-native-bits.bin')).arrayBuffer());
 for(let i=0;i<512;i++)if(actualBits[i]!==expectedBits[i])throw Error('Material RNG mismatch '+i+': '+actualBits[i]+' vs '+expectedBits[i]);
 let maxError=0;for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);maxError=Math.max(maxError,error);if(!Number.isFinite(actual[i])||error>(i%10<7?0:0.00003))throw Error('Material mismatch '+i+': '+actual[i]+' vs '+expected[i]);}
 overallMax=Math.max(overallMax,maxError);modes.push(objectHeap);
 }finally{arena?.dispose();runtime.destroyBuffer(out);runtime.destroyBuffer(bits);}
 }
 return {cases:1024,modes,materials:['lambertian','metal','dielectric'],virtualDispatch:true,randomStateExact:true,floatValues:10240,maxAbsoluteError:overallMax,fullPathtracerSupported:false};
}
