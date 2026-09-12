export async function checkChronoTypes(runtime){
 const source=await(await fetch(new URL('chrono-enums.cu',import.meta.url))).text(),reference=await(await fetch(new URL('../reports/chrono-enums-native.json',import.meta.url))).json();
 const kernel=await runtime.kernel(source,{entry:'chronoEnumValues',workgroupSize:[1]}),output=runtime.createBuffer(54*4);let compared=0;
 try{for(let i=0;i<4;i++){
  runtime.batch().dispatch(kernel.bind({output},{'constant.flags.periodic':!!(i&1),'constant.flags.invert':!!(i&2)}),[1]).submit();
  const actual=await runtime.read(output,Int32Array);if(reference[i]?.length!==actual.length)throw Error('Missing native enum reference');
  for(let j=0;j<actual.length;j++){if(actual[j]!==reference[i][j])throw Error('Chrono enum/boolean mismatch '+i+':'+j);compared++;}
 }}finally{runtime.destroyBuffer(output);}
 const fields=Array.from({length:120},(_,i)=>'float f'+i+';').join('');
 const large=await runtime.kernel('struct Parameters {'+fields+'bool enabled;};__constant__ Parameters params;__global__ void readLarge(float* output){output[0]=params.enabled?params.f119:params.f0;}',{entry:'readLarge',workgroupSize:[1]}),result=runtime.createBuffer(4);
 try{for(const enabled of [false,true]){runtime.batch().dispatch(large.bind({output:result},{'constant.params.enabled':enabled,'constant.params.f0':-4,'constant.params.f119':17}),[1]).submit();if((await runtime.read(result))[0]!==(enabled?17:-4))throw Error('Large constant record mismatch');compared++;}}finally{runtime.destroyBuffer(result);}
 return {originalScopedEnums:16,originalEnumValues:52,booleanCombinations:4,compared,nativeEnumsExact:true,largeConstantFields:121};
}
