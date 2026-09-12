export async function checkQuasirandom(runtime){
 const source=await(await fetch('/tests/quasirandom-kernels.cuh')).text(),table=new Uint32Array(await(await fetch('/reports/quasirandom-table.bin')).arrayBuffer());if(table.length!==93)throw Error('Invalid direction table');
 const k=await runtime.kernel(source,{entry:'quasirandomGeneratorKernel',workgroupSize:[128,3,1]}),scalars=Object.fromEntries(Array.from(table,(v,i)=>[`constant.c_Table[${Math.floor(i/31)}][${i%31}]`,v])),results=[];
 for(const [c,N,seed]of [[0,1048576,0],[1,1025,1],[2,1025,4294967280]]){
  const output=runtime.createBuffer(new Float32Array(N*3));try{runtime.batch().dispatch(k.bind({d_Output:output},{...scalars,N,seed}),[128,1,1]).submit();const actual=await runtime.read(output,Uint32Array),native=new Uint32Array(await(await fetch('/reports/quasirandom-'+c+'-native.bin')).arrayBuffer());if(native.length!==actual.length)throw Error('Native size mismatch');let mismatches=0;for(let i=0;i<actual.length;i++)if(actual[i]!==native[i])mismatches++;if(mismatches)throw Error(`Case ${c}: ${mismatches} native bit mismatches`);results.push({N,seed,components:actual.length,mismatches});}finally{runtime.destroyBuffer(output);}
 }
 return {passed:true,device:runtime.describe(),workgroupSize:[128,3,1],groups:[128,1,1],results};
}
