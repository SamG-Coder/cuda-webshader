export async function checkIntegerPowers(runtime){
 const load=p=>fetch(new URL(p,import.meta.url)),source=await(await load('integer-powers.cu')).text(),input=new Float32Array(await(await load('../reports/integer-powers-input.bin')).arrayBuffer()),expected=new Uint32Array(await(await load('../reports/integer-powers-native.bin')).arrayBuffer()),n=input.length/2;
 if(n!==264||expected.length!==n)throw Error('Incomplete integer-power native reference');
 const kernel=await runtime.kernel(source,{workgroupSize:[64]}),a=runtime.createBuffer(input),b=runtime.createBuffer(n*4);
 try{runtime.batch().dispatch(kernel.bind({input:a,output:b},{n}),[Math.ceil(n/64)]).submit();const actual=await runtime.read(b,Uint32Array),nan=v=>(v&0x7fffffff)>0x7f800000;let maxUlps=0;
 for(let i=0;i<n;i++){if(nan(actual[i])&&nan(expected[i]))continue;if(nan(actual[i])||nan(expected[i])||((actual[i]&0x7fffffff)===0x7f800000||(expected[i]&0x7fffffff)===0x7f800000)&&actual[i]!==expected[i])throw Error('Integer power nonfinite mismatch '+i);if((actual[i]&0x80000000)!==(expected[i]&0x80000000))throw Error('Integer power sign mismatch '+i);const delta=Math.abs(actual[i]-expected[i]);maxUlps=Math.max(maxUlps,delta);if(delta>1)throw Error('Integer power differs from native by '+delta+' ULP at '+i);}
 return {cases:n,maxUlps,negativeExponents:true,signedZeros:true,nonfiniteCases:true};
 }finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);}
}
