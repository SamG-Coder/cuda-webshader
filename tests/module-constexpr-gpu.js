export async function checkModuleConstexpr(runtime){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('module-constexpr.cu')).text(),reference=new Uint8Array(await(await load('../reports/module-constexpr-native.bin')).arrayBuffer());
 const kernel=await runtime.kernel(source,{workgroupSize:[64]}),output=runtime.createBuffer(65*8*4),flags=runtime.createBuffer(new Uint8Array(68).fill(0xa5));
 try{
  runtime.batch().dispatch(kernel.bind({output,flags},{}),[2]).submit();
  const a=new Uint8Array((await runtime.read(output,Uint32Array)).buffer),b=new Uint8Array((await runtime.read(flags,Uint32Array)).buffer),actual=new Uint8Array(a.length+b.length);actual.set(a);actual.set(b,a.length);
  if(actual.length!==reference.length||actual.some((v,i)=>v!==reference[i]))throw Error('Module constexpr/native CUDA mismatch');
  return {nativeExact:true,bytesCompared:actual.length,constantUniforms:kernel.artifact.metadata.scalars.length,volatileHelperFlags:65,paddingBytesUnchanged:3};
 }finally{runtime.destroyBuffer(output);runtime.destroyBuffer(flags);}
}
