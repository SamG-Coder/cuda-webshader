export async function checkShortValues(runtime) {
  const source=await(await fetch('/tests/short-values.cu')).text(),kernel=await runtime.kernel(source,{entry:'shortValues',workgroupSize:[1,1,1]}),response=await fetch('/reports/short-values-native.bin');
  if(!response.ok)throw Error('Missing native short capture');
  const native=new Int32Array(await response.arrayBuffer()),cases=[[32766,65534],[-32768,0],[-1,65535],[0,32768]];
  if(native.length!==cases.length*8)throw Error('Wrong native short capture size');
  for(const [scenario,[a,b]]of cases.entries()) {const out=runtime.createBuffer(32);try {
    runtime.batch().dispatch(kernel.bind({out},{a,b}),[1,1,1]).submit();
    const actual=await runtime.read(out,Int32Array);
    if(actual.some((v,i)=>v!==native[scenario*8+i]))throw Error('Short value differs from native case '+scenario);
  } finally {runtime.destroyBuffer(out);}}
  return {nativeExact:true,cases:cases.length,valuesPerCase:8};
}
