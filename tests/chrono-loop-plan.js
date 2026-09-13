import {createChronoRebuild} from './chrono-rebuild.js';
import {createChronoStep} from './chrono-step-plan.js';

// Host scheduling only. Every CUDA pass and diagnostic still executes.
// Validate the completed step while obtaining the next active count, so the
// next physics step cannot start until the previous diagnostics are checked.
export async function createChronoLoop(runtime,params,count,options={}){
 const rebuild=await createChronoRebuild(runtime,params,count,options);
 let step;
 try{step=await createChronoStep(runtime,params,{...options,deferDiagnostics:true});}
 catch(error){rebuild.dispose();throw error;}
 let prepared=null,steps=0,busy=false,disposed=false;
 const advance=async original=>{
  if(disposed||busy)throw Error('Chrono loop is disposed or already advancing');
  busy=true;
  try{
   prepared??=await rebuild(original,steps*.0001);
   const neighbors=prepared.neighborEntries;
   const status=await step(original,prepared);
   prepared.dispose();prepared=null;
   prepared=await rebuild(original,(steps+1)*.0001,{previousStatus:status});
   steps++;return {steps,neighbors};
  }catch(error){prepared?.dispose();prepared=null;throw error;}
  finally{busy=false;}
 };
 advance.dispose=()=>{if(busy)throw Error('Wait for the active Chrono step before disposing');if(disposed)return;disposed=true;prepared?.dispose();prepared=null;step.dispose();rebuild.dispose();};
 return advance;
}
