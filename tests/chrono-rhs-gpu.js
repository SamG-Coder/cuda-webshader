import {checkChronoAdami} from './chrono-adami-gpu.js';
export async function checkChronoRhs(runtime,{diagnostic=false,nativeBoundaryInput=false,afterForces}={}){
 const load=path=>fetch(new URL(path,import.meta.url)),source=await(await load('chrono-rhs.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),native=await(await load('../reports/chrono-rhs-native.bin')).arrayBuffer();
 const prerequisites=await checkRhsPrerequisites(runtime);
 const kernel=await runtime.kernel(source,{entry:'CfdCalcRHS_D',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});let result;
 const boundary=await checkChronoAdami(runtime,{afterBoundary:async({buffers:b,neighbors,rho,vel,count:n})=>{
  if(native.byteLength!==n*32+4)throw Error('Incomplete native RHS reference');
  const info=kernel.artifact.metadata.diagnostics,o={sortedDerivVelRho:runtime.createBuffer(n*16),sortedFreeSurfaceIdD:runtime.createBuffer(n*4),sortedPosDivergence:runtime.createBuffer(n*4),courantViscousTimeStep:runtime.createBuffer(n*4),accelerationTimeStep:runtime.createBuffer(n*4),error_flag:runtime.createBuffer(4),[info.buffer]:runtime.createBuffer((2+info.capacity*info.strideWords)*4)};
  try{
   let importedRho,importedVel;
   if(nativeBoundaryInput){const snapshot=await(await load('../reports/chrono-adami-native.bin')).arrayBuffer();importedRho=runtime.createBuffer(new Uint8Array(snapshot.slice(0,n*16)));importedVel=runtime.createBuffer(new Uint8Array(snapshot.slice(n*16,n*28)));o.importedRho=importedRho;o.importedVel=importedVel;}
   const before={...runtime.stats};runtime.batch().dispatch(kernel.bind(Object.fromEntries(Object.entries({...o,sortedPosRad:b.sortedPosRad,sortedVelMas:importedVel||vel,sortedRhoPreMu:importedRho||rho,numNeighborsPerPart:b.offsets,neighborList:neighbors}).filter(([key])=>!['importedRho','importedVel'].includes(key))),{...params,numActive:n}),[Math.ceil(n/128)]).submit();await runtime.idle();
   if(runtime.stats.readbackBytes!==before.readbackBytes||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in RHS');
   if(afterForces)await afterForces({buffers:b,neighbors,rho,vel,count:n,forces:o});
   const sections=[];let offset=0,compared=0,mismatches=0;
   for(const [name,words,integer]of [['sortedDerivVelRho',n*4,false],['sortedFreeSurfaceIdD',n,true],['sortedPosDivergence',n,false],['courantViscousTimeStep',n,false],['accelerationTimeStep',n,false],['error_flag',1,true]]){
    const Type=integer?Uint32Array:Float32Array,actual=await runtime.read(o[name],Type),expected=new Type(native,offset,words);let maximumError=0,failed=0;const examples=[],components=Array.from({length:name==='sortedDerivVelRho'?4:1},()=>({maximumError:0,index:0,actual:0,expected:0}));
    for(let i=0;i<words;i++){const a=actual[i],e=expected[i],error=a===e?0:Math.abs(a-e);if(error>maximumError)maximumError=error;if(error>components[i%components.length].maximumError)components[i%components.length]={maximumError:error,index:i,actual:a,expected:e};const absolute=name==='sortedDerivVelRho'?(i%4===3?1e-2:4e-4):1e-5,relative=name==='accelerationTimeStep'?3e-3:name==='sortedPosDivergence'?2e-5:2e-4;const valid=a===e||!integer&&name!=='courantViscousTimeStep'&&Number.isFinite(a)&&Number.isFinite(e)&&error<=absolute+relative*Math.abs(e);if(!valid){failed++;if(examples.length<4)examples.push({index:i,actual:a,expected:e,error});}compared++;}
    sections.push({name,words,maximumError,failed,examples,components});mismatches+=failed;offset+=words*4;
   }
   const messages=await runtime.read(o[info.buffer],Uint32Array);if(messages[0]||messages[1])throw Error('Original RHS reported invalid derivatives');
   result={tolerances:{acceleration:{absolute:4e-4,relative:2e-4},densityDerivative:{absolute:1e-2,relative:2e-4},positionDivergence:{absolute:1e-5,relative:2e-5},accelerationTimeStep:{absolute:1e-5,relative:3e-3},courantAndFlags:"exact"},markers:n,compared,mismatches,sections,nativeMatched:mismatches===0,connectedBoundary:!nativeBoundaryInput,intermediateReadbackBytes:0,cpuPhysics:false,fullSolverAdvanced:false};
   if(mismatches&&!diagnostic)throw Error(JSON.stringify(result));
  }finally{for(const buffer of Object.values(o))runtime.destroyBuffer(buffer);}
 }});
 if(!result)throw Error('RHS did not run');return {...result,boundaryCompared:boundary.compared,prerequisites};
}

export async function checkRhsPrerequisites(runtime){
 const source=await(await fetch(new URL('chrono-rhs-primitives.cu',import.meta.url))).text(),reference=new Uint32Array(await(await fetch(new URL('../reports/chrono-rhs-primitives-native.bin',import.meta.url))).arrayBuffer());
 const kernel=await runtime.kernel(source,{entry:'prerequisites',workgroupSize:[32]}),input=runtime.createBuffer(new Float32Array([1e8,1,-1e8,-2.75])),output=runtime.createBuffer(512);
 try{runtime.batch().dispatch(kernel.bind({input,output},{}),[1]).submit();const actual=await runtime.read(output,Uint32Array);if(actual.length!==reference.length||actual.some((v,i)=>v!==reference[i]))throw Error('Chrono force prerequisite mismatch');return {nativeExact:true,compared:actual.length,longExpressionTerms:900,localArraySizes:[8,12]};}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
}
