export async function checkSobol(runtime){
 const source=await(await fetch('/tests/sobol-kernels.cuh')).text(),k=await runtime.kernel(source,{entry:'sobolGPU_kernel',workgroupSize:[64,1,1]}),results=[];
 for(const [c,N,D,gx]of [[0,100000,100,512],[1,1025,3,32],[2,2000,512,1]]){
  const directions=new Uint32Array(await(await fetch('/reports/sobol-'+c+'-directions.bin')).arrayBuffer());if(directions.length!==D*32)throw Error('Invalid direction table');const dd=runtime.createBuffer(directions),out=runtime.createBuffer(new Float32Array(N*D));
  try{runtime.batch().dispatch(k.bind({d_directions:dd,d_output:out},{n_vectors:N,n_dimensions:D}),[gx,D,1]).submit();const actual=await runtime.read(out,Uint32Array),native=new Uint32Array(await(await fetch('/reports/sobol-'+c+'-native.bin')).arrayBuffer());if(native.length!==actual.length)throw Error('Native size mismatch');let mismatches=0;for(let i=0;i<actual.length;i++)if(actual[i]!==native[i])mismatches++;if(mismatches)throw Error(`Case ${c}: ${mismatches} native bit mismatches`);results.push({vectors:N,dimensions:D,groups:[gx,D,1],components:actual.length,mismatches});}finally{runtime.destroyBuffer(dd);runtime.destroyBuffer(out);}
 }
 return {passed:true,device:runtime.describe(),workgroupSize:[64,1,1],sharedBytes:k.artifact.metadata.workgroupStorageBytes,results};
}
