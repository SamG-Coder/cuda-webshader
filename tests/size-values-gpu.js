export async function checkSizeValues(runtime){
 const source=await(await fetch('/tests/size-values-kernel.cuh')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1,1,1]});
 const words=runtime.createBuffer(new Uint32Array(5)),values=runtime.createBuffer(new Float32Array(2)),expectedWords=new Uint32Array(await(await fetch('/reports/size-values-words.bin')).arrayBuffer()),expectedValues=new Float32Array(await(await fetch('/reports/size-values-floats.bin')).arrayBuffer());
 const cases=[[192,7],[0xffffffff,2],[12345,-1],[0,0],[0xffffffff,2147483647]];
 try{
  for(let i=0;i<cases.length;i++){const [pitch,row]=cases[i];runtime.batch().dispatch(kernel.bind({words,values},{pitch,row}),[1,1,1]).submit();const w=await runtime.read(words,Uint32Array),v=await runtime.read(values);if(w.some((n,j)=>n!==expectedWords[i*5+j])||v.some((n,j)=>n!==expectedValues[i*2+j]))throw Error('Native size_t arithmetic mismatch '+i);}
  return {cases,components:35,nativeExact:true,launchValueBits:32,arithmeticBits:64};
 }finally{runtime.destroyBuffer(words);runtime.destroyBuffer(values);}
}
