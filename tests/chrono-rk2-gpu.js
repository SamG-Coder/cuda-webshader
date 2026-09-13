import {checkChronoRhs} from './chrono-rhs-gpu.js';
export async function checkChronoRk2(runtime){
 const load=async p=>await(await fetch(new URL(p,import.meta.url))).text(),params=JSON.parse(await load('../reports/chrono-params.json'));
 const integration=await load('chrono-rk2.cu'),shiftSource=await load('chrono-shifting.cu'),rhsSource=await load('chrono-rhs.cu'),bcSource=await load('chrono-adami.cu');
 const make=(source,entry)=>runtime.kernel(source,{entry,defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
 const euler=await make(integration,'EulerStep_D'),periodic=await make(integration,'ApplyPeriodicBoundaryY_D'),shift=await make(shiftSource,'Calc_Shifting_D<ShiftingMethod::XSPH>'),bc=await make(bcSource,'CfdAdamiBC_D'),rhs=await make(rhsSource,'CfdCalcRHS_D');
 let result;
 await checkChronoRhs(runtime,{afterForces:async({buffers:b,neighbors,rho,vel,count:n,forces:f})=>{
  const owned=[],alloc=size=>{const v=runtime.createBuffer(size);owned.push(v);return v;},state=()=>({pos:alloc(n*16),vel:alloc(n*12),rho:alloc(n*16)}),y=state(),tmp=state(),half=state(),flag=alloc(4),shifting=alloc(n*12),acc=alloc(n*12),tauA=alloc(12),tauB=alloc(12),pc=alloc(12),dtauA=alloc(12),dtauB=alloc(12),deriv=alloc(n*16),surface=alloc(n*4),divergence=alloc(n*4),courant=alloc(n*4),acceleration=alloc(n*4),diags=new Map();
  for(const k of [euler,periodic,shift,bc,rhs])if(k.artifact.metadata.diagnostics){const d=k.artifact.metadata.diagnostics;diags.set(k,alloc((2+d.capacity*d.strideWords)*4));}
  const bind=(k,data,extra={})=>{const d=k.artifact.metadata.diagnostics;if(d)data={...data,[d.buffer]:diags.get(k)};return k.bind(Object.fromEntries(k.artifact.metadata.bindings.map(x=>{if(!data[x.name])throw Error('Missing '+x.name);return [x.name,data[x.name]]})),{...params,numActive:n,...extra});};
  const shiftBind=(s,div)=>bind(shift,{vel_XSPH_Sorted_D:shifting,sortedPosRad:s.pos,sortedVelMas:s.vel,sortedRhoPreMu:s.rho,numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosDivergence:div,error_flag:flag});
  const eulerBind=(s,d,surf,dt)=>bind(euler,{posRadD:s.pos,velMasD:s.vel,rhoPresMuD:s.rho,tauXxYyZzD:tauA,tauXyXzYzD:tauB,pcEvSvD:pc,vel_XSPH_D:shifting,derivVelRhoD:d,derivTauXxYyZzD:dtauA,derivTauXyXzYzD:dtauB,freeSurfaceIdD:surf,activityIdentifierSortedD:b.sortedActivity,error_flag:flag},{dT:dt});
  const periodicBind=s=>bind(periodic,{posRadD:s.pos,rhoPresMuD:s.rho});
  const copy=(src,dests)=>{const enc=runtime.device.createCommandEncoder();for(const dst of dests)for(const [name,bytes]of [['pos',n*16],['vel',n*12],['rho',n*16]])enc.copyBufferToBuffer(src[name].gpuBuffer,0,dst[name].gpuBuffer,0,bytes);runtime.device.queue.submit([enc.finish()]);};
  try{
   const before={...runtime.stats},grid=[Math.ceil(n/128)],dt=params['constant.paramsD.dT'];copy({pos:b.sortedPosRad,vel,rho},[y,tmp]);
   runtime.batch().dispatch(shiftBind(y,f.sortedPosDivergence),grid).dispatch(eulerBind(tmp,f.sortedDerivVelRho,f.sortedFreeSurfaceIdD,dt/2),grid).dispatch(periodicBind(tmp),grid).submit();copy(tmp,[half]);
   runtime.batch().dispatch(bind(bc,{numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosRadD:tmp.pos,bceAcc:acc,sortedRhoPresMuD:tmp.rho,sortedVelMasD:tmp.vel,error_flag:flag}),grid)
    .dispatch(bind(rhs,{sortedDerivVelRho:deriv,sortedPosRad:tmp.pos,sortedVelMas:tmp.vel,sortedRhoPreMu:tmp.rho,numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedFreeSurfaceIdD:surface,sortedPosDivergence:divergence,courantViscousTimeStep:courant,accelerationTimeStep:acceleration,error_flag:flag}),grid)
    .dispatch(shiftBind(tmp,divergence),grid).dispatch(eulerBind(y,deriv,surface,dt),grid).dispatch(periodicBind(y),grid).submit();await runtime.idle();
   if(runtime.stats.readbackBytes!==before.readbackBytes||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('CPU transfer during RK2');
   const stages=[];
   for(const [state,path]of [[half,'chrono-rk2-half-native.bin'],[y,'chrono-rk2-native.bin']]){const native=await(await fetch(new URL('../reports/'+path,import.meta.url))).arrayBuffer();if(native.byteLength!==n*44+4||new Uint32Array(native,n*44,1)[0]!==0)throw Error('Invalid native RK2 reference');let offset=0;const sections=[];
    for(const [name,words]of [['pos',n*4],['vel',n*3],['rho',n*4]]){const actual=await runtime.read(state[name],Float32Array),expected=new Float32Array(native,offset,words);let max=0,mismatches=0,nonzero=0;const components=Array.from({length:name==='vel'?3:4},()=>({maximumError:0,unequal:0}));const examples=[];for(let i=0;i<words;i++){const err=Math.abs(actual[i]-expected[i]);max=Math.max(max,err);const component=components[i%components.length];component.maximumError=Math.max(component.maximumError,err);if(actual[i]!==expected[i])component.unequal++;if(actual[i]!==0)nonzero++;const abs=name==='vel'?4e-7:name==='rho'?2e-4:1e-6,rel=name==='rho'?2e-5:2e-4;if(!Number.isFinite(actual[i])||!Number.isFinite(expected[i])||err>abs+rel*Math.abs(expected[i])||name==='pos'&&i%4===3&&err!==0||name==='rho'&&i%4>=2&&err!==0){mismatches++;if(examples.length<4)examples.push({i,actual:actual[i],expected:expected[i],err});}}sections.push({name,max,mismatches,nonzero,examples,components});offset+=words*4;}
    stages.push({path,sections});
   }
   const errors=await runtime.read(flag,Uint32Array),messages=[];for(const d of diags.values()){const a=await runtime.read(d,Uint32Array);messages.push([a[0],a[1]]);}
   const failed=stages.some(s=>s.sections.some(v=>v.mismatches))||errors[0]||messages.some(v=>v[0]||v[1]);
   result={stages,errors:[...errors],messages,fullStepDispatched:true,rk2Steps:1,nativeMatched:!failed,trajectoryValidated:false,compared:2*(n*11+1),dt,intermediateReadbackBytes:0,tolerances:{position:{absolute:1e-6,relative:2e-4},velocity:{absolute:4e-7,relative:2e-4},properties:{absolute:2e-4,relative:2e-5},radiusViscosityMarkerTypeAndFlags:'exact'}};
   if(failed)throw Error(JSON.stringify(result));
  }finally{for(const o of owned)runtime.destroyBuffer(o);}
 }});
 if(!result)throw Error('RK2 did not execute');return result;
}
