import {createChronoLoop} from './chrono-loop-plan.js';
import {createChronoRebuild} from './chrono-rebuild.js';
import {createChronoStep} from './chrono-step-plan.js';

export async function checkChronoLoop(runtime){
 const params=await(await fetch('/reports/chrono-params.json')).json(),a={},b={};
 let loop,rebuild,step;
 const status=runtime.createBuffer(new Uint32Array([1]));
 try{
  for(const [key,path]of Object.entries({pos:'chrono-search-input.bin',vel:'chrono-marker-velocities.bin',rho:'chrono-marker-rhopremu.bin'})){
   const data=new Uint8Array(await(await fetch('/reports/'+path)).arrayBuffer());a[key]=runtime.createBuffer(data);b[key]=runtime.createBuffer(data);
  }
  loop=await createChronoLoop(runtime,params,30327);
  rebuild=await createChronoRebuild(runtime,params,30327);step=await createChronoStep(runtime,params);
  for(let i=0;i<3;i++){
   const pending=loop(a);let rejected=false;try{await loop(a);}catch{rejected=true;}
   const result=await pending;if(!rejected)throw Error('Concurrent physics steps accepted');
   const prepared=await rebuild(b,i*.0001);
   try{if(result.neighbors!==prepared.neighborEntries)throw Error('Neighbour counts differ');await step(b,prepared);}finally{prepared.dispose();}
  }
  for(const key of Object.keys(a)){
   const actual=await runtime.read(a[key],Uint32Array),expected=await runtime.read(b[key],Uint32Array);
   if(actual.some((value,i)=>value!==expected[i]))throw Error('Pipelined state differs: '+key);
  }
  let rejected=false;try{await rebuild(b,.0003,{previousStatus:status});}catch(error){rejected=error.message.includes('Chrono integration diagnostic');}
  if(!rejected)throw Error('Previous step error was not rejected');
  return {steps:3,exactState:true,exactNeighbors:true,concurrentStepRejected:true,previousDiagnosticRejected:true};
 }finally{loop?.dispose();step?.dispose();rebuild?.dispose();for(const resource of [...Object.values(a),...Object.values(b),status])runtime.destroyBuffer(resource);}
}
