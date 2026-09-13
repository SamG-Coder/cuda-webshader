import {createChronoRebuild} from './chrono-rebuild.js';
import {createChronoStep} from './chrono-step-plan.js';

export async function checkChronoBindings(runtime){
 const params=await(await fetch('/reports/chrono-params.json')).json(),dt=params['constant.paramsD.dT'],initial={},a={},b={};
 let rebuild,step;
 try{
  for(const [key,path]of Object.entries({pos:'chrono-search-input.bin',vel:'chrono-marker-velocities.bin',rho:'chrono-marker-rhopremu.bin'})){
   initial[key]=new Float32Array(await(await fetch('/reports/'+path)).arrayBuffer());a[key]=runtime.createBuffer(initial[key]);b[key]=runtime.createBuffer(initial[key]);
  }
  rebuild=await createChronoRebuild(runtime,params,30327);step=await createChronoStep(runtime,params);
  const counts=[];
  for(const phase of ['full','changed-parameters','partial','full','empty','full']){
   params['constant.paramsD.dT']=phase==='changed-parameters'?dt*2:dt;
   const pos=initial.pos.slice(),rho=initial.rho.slice();
   if(phase==='partial'||phase==='empty')for(let i=0;i<(phase==='partial'?8000:30327);i++){pos[i*4]=1e6;if(phase==='empty')rho[i*4+3]=-1;}
   for(const state of [a,b]){runtime.write(state.pos,pos);runtime.write(state.rho,rho);runtime.write(state.vel,initial.vel);}
   let freshRebuild,freshStep,x,y;
   try{
    freshRebuild=await createChronoRebuild(runtime,params,30327);freshStep=await createChronoStep(runtime,params);
    x=await rebuild(a,0);y=await freshRebuild(b,0);counts.push(x.count);
    if(x.count!==y.count||x.neighborEntries!==y.neighborEntries)throw Error('Cached preparation differs');
    await step(a,x);await freshStep(b,y);
    for(const key of Object.keys(a)){
     const actual=await runtime.read(a[key],Uint32Array),expected=await runtime.read(b[key],Uint32Array);
     if(actual.some((v,i)=>v!==expected[i]))throw Error('Cached binding changed '+phase+' '+key);
    }
   }finally{x?.dispose();y?.dispose();freshStep?.dispose();freshRebuild?.dispose();}
  }
  if(String(counts)!=='30327,30327,22327,30327,0,30327')throw Error('Binding test did not exercise count changes');
  return {counts,exactState:true,parameterUpdates:true,resizedBuffers:true,resumeAfterEmpty:true};
 }finally{step?.dispose();rebuild?.dispose();for(const resource of [...Object.values(a),...Object.values(b)])runtime.destroyBuffer(resource);}
}
