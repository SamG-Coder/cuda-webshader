export async function checkSobelNeighborhoods(runtime) {
  const source=(await Promise.all(['sobel-compute.cuh','sobel-neighborhoods.cu'].map(async f=>(await fetch('/tests/'+f)).text()))).join('\n');
  const kernel=await runtime.kernel(source,{entry:'sobelNeighborhoods',workgroupSize:[128,1,1]});
  const n=4099,size=4120,input=Uint8Array.from({length:n*9},(_,i)=>(i*37+Math.floor(i/9)*13)&255),results=[];
  for(const [scenario,scale] of [-1,0,.25,1,4].entries()) {
    const src=runtime.createBuffer(input),dst=runtime.createBuffer(new Uint8Array(size).fill(165));
    try {
      runtime.batch().dispatch(kernel.bind({input:src,output:dst},{n,scale}),[Math.ceil(n/128),1,1]).submit();
      const actual=new Uint8Array((await runtime.read(dst,Uint32Array)).buffer),response=await fetch('/reports/sobel-neighborhoods-native-'+scenario+'.bin');
      if(!response.ok)throw Error('Missing native Sobel capture');
      const native=new Uint8Array(await response.arrayBuffer());
      if(actual.length!==native.length||actual.some((v,i)=>v!==native[i]))throw Error('Original ComputeSobel differs from native at scale '+scale);
      results.push({scale,neighborhoods:n,guardBytes:size-n,nativeExact:true});
    } finally {runtime.destroyBuffer(src);runtime.destroyBuffer(dst);}
  }
  return {originalHelper:true,results};
}
