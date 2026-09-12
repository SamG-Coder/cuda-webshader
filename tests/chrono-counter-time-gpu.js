export async function checkChronoCounterTime(runtime){
 const source=await(await fetch(new URL('chrono-counter-time.cu',import.meta.url))).text();
 const cases=await(await fetch(new URL('../reports/chrono-counter-time-native.json',import.meta.url))).json();
 if(cases.length!==12||cases.some(c=>c.output.length!==96))throw Error('Incomplete native counter/time reference');
 const kernel=await runtime.kernel(source,{entry:'counterTimeProbe',workgroupSize:[1]}),output=runtime.createBuffer(96*4);
 let compared=0;
 try{for(const [index,c] of cases.entries()){
  runtime.batch().dispatch(kernel.bind({output},c.scalars),[1]).submit();
  const actual=await runtime.read(output,Uint32Array);
  for(let i=0;i<actual.length;i++){if(actual[i]!==c.output[i])throw Error(`Counter/time case ${index}, value ${i}: ${actual[i]} != ${c.output[i]}`);compared++;}
 }}finally{runtime.destroyBuffer(output);}
 return {originalCounterFields:22,nativeCounterBytes:176,cases:12,compared,nativeExact:true,counterHighWordsPreserved:true,doubleTimePreserved:true};
}
