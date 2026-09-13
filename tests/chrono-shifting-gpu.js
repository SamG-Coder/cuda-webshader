import {checkChronoRhs} from './chrono-rhs-gpu.js';
export async function checkChronoShifting(runtime){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-shifting.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),native=await(await load('../reports/chrono-shifting-native.bin')).arrayBuffer();
 if(params['constant.paramsD.shifting_method']!==2)throw Error('Reference must use original XSPH configuration');
 const kernel=await runtime.kernel(source,{entry:'Calc_Shifting_D<ShiftingMethod::XSPH>',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
 let result;
 const forces=await checkChronoRhs(runtime,{afterForces:async({buffers:b,neighbors,rho,vel,count:n,forces:f})=>{
  if(native.byteLength!==2*(n*12+4))throw Error('Incomplete native shifting reference');
  const info=kernel.artifact.metadata.diagnostics,o={vel_XSPH_Sorted_D:runtime.createBuffer(n*12),error_flag:runtime.createBuffer(4),[info.buffer]:runtime.createBuffer((2+info.capacity*info.strideWords)*4)};
  try{
   const cases=[];
   for(let mode=0;mode<2;mode++){
   let velocity=vel;
   if(mode){const values=new Float32Array(n*3);for(let i=0;i<n;i++){values[i*3]=(i%17-8)*0.125;values[i*3+1]=(i%13-6)*0.0625;values[i*3+2]=(i%11-5)*0.25;}o.syntheticVelocity=runtime.createBuffer(values);velocity=o.syntheticVelocity;}
   const bindings=Object.fromEntries(Object.entries(o).filter(([key])=>key!=='syntheticVelocity'));
   const before={...runtime.stats};
   runtime.batch().dispatch(kernel.bind({...bindings,sortedPosRad:b.sortedPosRad,sortedVelMas:velocity,sortedRhoPreMu:rho,numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosDivergence:f.sortedPosDivergence},{...params,numActive:n}),[Math.ceil(n/128)]).submit();await runtime.idle();
   if(runtime.stats.readbackBytes!==before.readbackBytes||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer during shifting');
   const actual=await runtime.read(o.vel_XSPH_Sorted_D,Float32Array),expected=new Float32Array(native,mode*(n*12+4),n*3);let maximumError=0,nonzero=0;
   for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);if(!Number.isFinite(actual[i])||error>1e-5+2e-4*Math.abs(expected[i]))throw Error('Shifting mismatch at '+i);maximumError=Math.max(maximumError,error);if(actual[i]!==0)nonzero++;}
   const flags=await runtime.read(o.error_flag,Uint32Array),messages=await runtime.read(o[info.buffer],Uint32Array);
   if(flags[0]!==new Uint32Array(native,mode*(n*12+4)+n*12,1)[0]||flags[0]||messages[0]||messages[1])throw Error('Original shifting reported invalid output');
   if(mode&&nonzero<1000)throw Error('Nonzero velocity test did not exercise shifting');
   cases.push({mode:mode?'synthetic velocity validation':'native initialized state',nonzero,maximumError});
   result={markers:n,compared:2*(n*3+1),maximumError,nonzero,nativeMatched:true,method:'XSPH',initializedStateAndSyntheticVelocity:true,tolerances:{absolute:1e-5,relative:2e-4},intermediateReadbackBytes:0,cpuPhysics:false,fullSolverAdvanced:false,cases};
   }
  }finally{for(const buffer of Object.values(o))runtime.destroyBuffer(buffer);}
 }});
 if(!result)throw Error('Shifting did not run');return {...result,forcesCompared:forces.compared};
}
