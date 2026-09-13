import {checkChronoSelected} from './chrono-selected-gpu.js';
export async function checkChronoAdami(runtime,{diagnostic=false}={}){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-adami.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),native=await(await load('../reports/chrono-adami-native.bin')).arrayBuffer();
 const kernel=await runtime.kernel(source,{entry:'CfdAdamiBC_D',workgroupSize:[128]});let result;
 const preparation=await checkChronoSelected(runtime,{afterNeighbors:async({caseInfo:c,buffers:b,neighbors,count:n})=>{
  if(c.mode!==0)return;if(n!==30327||native.byteLength!==n*28+4)throw Error('Unexpected Adami reference');
  const rho=runtime.createBuffer(n*16),vel=runtime.createBuffer(n*12),acc=runtime.createBuffer(n*12),flag=runtime.createBuffer(4);
  try{
   const before={...runtime.stats};runtime.batch().copy(b.sortedRho,rho).copy(b.sortedVel,vel).dispatch(kernel.bind({numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosRadD:b.sortedPosRad,bceAcc:acc,sortedRhoPresMuD:rho,sortedVelMasD:vel,error_flag:flag},{...params,numActive:n}),[Math.ceil(n/128)]).submit();await runtime.idle();
   if(runtime.stats.readbackBytes!==before.readbackBytes||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer during boundary calculation');
   let offset=0,compared=0,maxAbsoluteError=0,maxScaledError=0,mismatches=0;const firstErrors=[];
   for(const [buffer,words]of [[rho,n*4],[vel,n*3]]){const actual=await runtime.read(buffer,Float32Array),expected=new Float32Array(native,offset,words);
    for(let i=0;i<words;i++){const error=Math.abs(actual[i]-expected[i]),pressure=offset===0&&i%4===1,exact=offset===0&&(expected[i-i%4+3]<-.5||i%4===2||i%4===3),absolute=pressure?2e-4:1e-5;if(!Number.isFinite(actual[i])||!Number.isFinite(expected[i])||(exact?actual[i]!==expected[i]:error>absolute+2e-5*Math.abs(expected[i]))){mismatches++;if(firstErrors.length<8)firstErrors.push({word:offset/4+i,actual:actual[i],expected:expected[i],error});}maxAbsoluteError=Math.max(maxAbsoluteError,error);maxScaledError=Math.max(maxScaledError,error/Math.max(1,Math.abs(expected[i])));compared++;}offset+=words*4;
   }
   if((await runtime.read(flag,Uint32Array))[0]!==new Uint32Array(native,offset,1)[0])throw Error('Adami error flag mismatch');
   if(mismatches&&!diagnostic)throw Error(JSON.stringify({mismatches,maxAbsoluteError,maxScaledError,firstErrors}));
   result={markers:n,mismatches,firstErrors,compared,maxAbsoluteError,maxScaledError,absoluteTolerance:1e-5,pressureAbsoluteTolerance:2e-4,relativeTolerance:2e-5,nativeMatched:mismatches===0,originalBoundaryKernel:true,connectedGpuPreparation:true,boundaryIntermediateReadbackBytes:0,cpuPhysics:false,fullSolverAdvanced:false};
  }finally{for(const buffer of [rho,vel,acc,flag])runtime.destroyBuffer(buffer);}
 }});
 if(!result)throw Error('Boundary case did not execute');return {...result,preparationCompared:preparation.compared};
}
