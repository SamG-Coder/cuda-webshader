import {createChronoRebuild} from './chrono-rebuild.js';

export async function checkChronoActiveCount(runtime){
 const params=await(await fetch('/reports/chrono-params.json')).json(),initial={},a={},b={};
 let cpu,gpu;
 try{
  for(const [key,path]of Object.entries({pos:'chrono-search-input.bin',vel:'chrono-marker-velocities.bin',rho:'chrono-marker-rhopremu.bin'})){
   initial[key]=new Float32Array(await(await fetch('/reports/'+path)).arrayBuffer());a[key]=runtime.createBuffer(initial[key]);b[key]=runtime.createBuffer(initial[key]);
  }
  cpu=await createChronoRebuild(runtime,params,30327);gpu=await createChronoRebuild(runtime,params,30327,{gpuActiveCount:true});
  const results=[];
  for(const empty of [false,true]){
   const pos=initial.pos.slice(),rho=initial.rho.slice();
   for(let i=0;i<30327;i++)if(empty||i<8000){pos[i*4]=1e6;if(empty)rho[i*4+3]=-1;}
   for(const state of [a,b]){runtime.write(state.pos,pos);runtime.write(state.rho,rho);runtime.write(state.vel,initial.vel);}
   let expected,actual;
   try{
    expected=await cpu(a,0);actual=await gpu(b,0);
    if(actual.count!==expected.count||actual.neighborEntries!==expected.neighborEntries||actual.count>=30327||(empty&&actual.count!==0))throw Error('GPU active count differs');
    for(const [key,bytes]of [['indices',actual.count*4],['sortedPosRad',actual.count*16],['sortedVel',actual.count*12],['sortedRho',actual.count*16],['offsets',(actual.count+1)*4]]){
     const x=await runtime.read(actual.buffers[key],Uint32Array,bytes),y=await runtime.read(expected.buffers[key],Uint32Array,bytes);if(x.some((v,i)=>v!==y[i]))throw Error('Active subset differs: '+key);
    }
    const x=await runtime.read(actual.neighbors,Uint32Array,actual.neighborEntries*4),y=await runtime.read(expected.neighbors,Uint32Array,expected.neighborEntries*4);if(x.some((v,i)=>v!==y[i]))throw Error('Active neighbour list differs');
    results.push({empty,count:actual.count,neighbors:actual.neighborEntries,exact:true});
   }finally{actual?.dispose();expected?.dispose();}
  }
  const status=runtime.createBuffer(new Uint32Array([1]));
  try{let rejected=false;try{await gpu(b,0,{previousStatus:status});}catch(error){rejected=error.message.includes('Chrono integration diagnostic');}if(!rejected)throw Error('GPU count path ignored previous diagnostic');}finally{runtime.destroyBuffer(status);}
  return {results,previousDiagnosticRejected:true};
 }finally{cpu?.dispose();gpu?.dispose();for(const resource of [...Object.values(a),...Object.values(b)])runtime.destroyBuffer(resource);}
}
