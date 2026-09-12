import {decodePGM} from '../src/sandbox/pgm.js';
export async function checkSobelShared(runtime){
 const source=await(await fetch('/tests/sobel-shared-kernel.cuh')).text(),results=[];
 for(let scenario=0;scenario<4;scenario++){
  let w=scenario===0?32:64,h=scenario===0?17:65,data;
  if(scenario>=2){const image=decodePGM(new Uint8Array(await(await fetch('/showcases/sobel/teapot.pgm')).arrayBuffer()));w=image.width;h=image.height;data=Uint8Array.from(image.data,v=>Math.round(v*255));}else data=Uint8Array.from({length:w*h},(_,i)=>(i*37+Math.floor(i/w)*13)&255);
  const Pitch=w+(scenario<2?4:0),size=Math.floor((Pitch*h+19)/4)*4,scale=scenario===3?.25:1,tex=runtime.createByteTexture2D(data,{width:w,height:h});
  try{for(const [pass,entry]of ['SobelShared'].entries()){
   const kernel=await runtime.kernel(source,{entry,workgroupSize:[16,4,1],sharedMemoryBytes:2304}),dst=runtime.createBuffer(new Uint8Array(size).fill(165));
   try{runtime.batch().dispatch(kernel.bind({pSobelOriginal:dst,tex},{SobelPitch:Pitch,BlockWidth:80,SharedPitch:384,w,h,fScale:scale}),[Math.ceil(w/320),Math.ceil(h/4),1]).submit();const actual=new Uint8Array((await runtime.read(dst,Uint32Array)).buffer),response=await fetch('/reports/sobel-shared-native-'+scenario+'-'+pass+'.bin');if(!response.ok)throw Error('Missing native image');const native=new Uint8Array(await response.arrayBuffer());if(native.length!==actual.length||actual.some((v,i)=>v!==native[i]))throw Error('Sobel image mismatch '+scenario+' '+entry);results.push({entry,width:w,height:h,pitch:Pitch,scale,nativeExact:true,guardsIntact:true});}finally{runtime.destroyBuffer(dst);}
  }}finally{runtime.destroyTexture(tex);}
 }return {results};
}
