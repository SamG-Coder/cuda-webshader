export async function checkBinomial(runtime){
 if(runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
 const source=await(await fetch(new URL('./binomial-kernels.cuh',import.meta.url))).text(),prepared=new Uint32Array(await(await fetch(new URL('../reports/binomial-options-prepared.bin',import.meta.url))).arrayBuffer()),expected=new Float32Array(await(await fetch(new URL('../reports/binomial-options-native.bin',import.meta.url))).arrayBuffer());
 const kernel=await runtime.kernel(source,{entry:'binomialOptionsKernel',workgroupSize:[128,1,1]}),input=runtime.createBuffer(prepared),output=runtime.createBuffer(4096);
 try{runtime.batch().dispatch(kernel.bind({d_OptionData:input,d_CallValue:output},{}),[1024,1,1]).submit();const actual=await runtime.read(output);let delta=0,reference=0,maxError=0;
 for(let i=0;i<1024;i++){if(!Number.isFinite(actual[i])||actual[i]<0)throw Error('Invalid option '+i);const error=Math.abs(actual[i]-expected[i]);delta+=error;reference+=Math.abs(expected[i]);maxError=Math.max(maxError,error);if(error>2e-4+Math.abs(expected[i])*5e-4)throw Error('Native option mismatch '+i+': '+actual[i]+' vs '+expected[i]);}
 const l1=delta/reference;if(l1>5e-4)throw Error('Native L1 mismatch '+l1);
 return {passed:true,device:runtime.describe(),options:1024,steps:2048,l1,maxAbsoluteError:maxError,allOptionsCompared:true,softwareAdapterRequested:false};
 }finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
}
