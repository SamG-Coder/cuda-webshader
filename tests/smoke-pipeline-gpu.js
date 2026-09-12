import {executePipeline} from '../src/sandbox/pipeline.js';
import {compile} from '../src/compiler/compiler.js';
export async function checkSmokePipeline(runtime){
 const source=await(await fetch('/tests/smoke-integration-kernel.cuh')).text()+await(await fetch('/tests/smoke-pipeline-adapter.cuh')).text();
 const count=257,buffers={};
 for(const name of ['positions','velocities','oldPositions','oldVelocities'])buffers[name]={type:'vec4<f32>',records:count,fill:name.startsWith('old')?'binary-f32':'zero',...(name.startsWith('old')?{source:'/reports/smoke-integration-input-'+(name==='oldPositions'?'positions':'velocities')+'.bin'}:{})};
 buffers.depth={type:'f32',records:count,fill:'zero'};buffers.indices={type:'u32',records:count,fill:'ramp'};
 const constants=Object.fromEntries(Object.entries({'gravity.x':0,'gravity.y':.0001,'gravity.z':0,globalDamping:.99,noiseFreq:.1,noiseAmp:.001,'noiseSpeed.x':.01,'noiseSpeed.y':.02,'noiseSpeed.z':-.01,time:{time:true}}).map(([k,v])=>['constant.cudaParams.'+k,v]));
 const depthScalars={count,'sortVector.x':.25,'sortVector.y':-.5,'sortVector.z':1};
 const dispatch=(entry,bindings,scalars)=>({entry,bindings,scalars,block:[128,1,1],groups:[3,1,1]});
 const plan={buffers,textures:{noise:{kind:'volume-float4',dimensions:[64,64,64],source:'/reports/smoke-noise-input.bin',filter:'linear',addressMode:'repeat'}},steps:[
  dispatch('integrate_functor',{tuple0:'positions',tuple1:'velocities',tuple2:'oldPositions',tuple3:'oldVelocities',noiseTex:'noise'},{...constants,deltaTime:.5,count}),
  dispatch('calcDepth_functor',{tuple0:'positions',tuple1:'depth'},depthScalars),
  dispatch('smokeIndices',{indices:'indices'},{count}),
  {sortPairs:{keys:'depth',values:'indices',count,keyType:'f32'}},
  {copyBuffer:{source:'positions',target:'oldPositions',records:count}},
  {copyBuffer:{source:'velocities',target:'oldVelocities',records:count}}
 ],preview:{kind:'particles',positions:'positions',count,radius:.02}};
 const resources=[],textureResources=[],artifacts=[],comparisons=[];
 try{
  const before=runtime.stats.readbackBytes;
  const result=await executePipeline({plan,source,runtime,compiler:{compile:async(s,o)=>({artifact:compile(s,o)})},rootArtifact:compile(source,{entry:'integrate_functor',workgroupSize:[128,1,1]}),resources,textureResources,onArtifact:a=>artifacts.push(a.name)});
  if(runtime.stats.readbackBytes!==before)throw Error('Pipeline initialization readback');
  for(let frame=1;frame<=64;frame++){
   if(frame===32)Object.assign(depthScalars,{'sortVector.x':-1,'sortVector.y':.125,'sortVector.z':.5});
   if(frame>1){const before=runtime.stats.readbackBytes;await result.stepFrame((frame-1)*.5);if(runtime.stats.readbackBytes!==before)throw Error('Smoke pipeline intermediate readback');}
   if(![1,8,32,64].includes(frame))continue;
   for(const name of ['positions','velocities']){
    const actual=await runtime.read(result.buffers[name]),expected=new Float32Array(await(await fetch('/reports/smoke-integration-'+frame+'-'+name+'.bin')).arrayBuffer());
    if(actual.length!==expected.length||actual.some((v,i)=>!Number.isFinite(v)||v!==expected[i]))throw Error('Smoke pipeline native mismatch '+name+' frame '+frame);
   }
   const keys=await runtime.read(result.buffers.depth,Uint32Array),indices=await runtime.read(result.buffers.indices,Uint32Array),expected=new Uint32Array(await(await fetch('/reports/smoke-sort-'+frame+'-native.bin')).arrayBuffer());
   if(expected.length!==count*2||keys.some((v,i)=>v!==expected[i])||indices.some((v,i)=>v!==expected[count+i]))throw Error('Pipeline sorted depth mismatch');
   comparisons.push({frame,positionsExact:true,velocitiesExact:true,sortedKeysExact:true,indicesExact:true});
  }
  if(result.steps!==64||!['sortPrepare','sortStage','sortFinish'].every(n=>artifacts.includes(n)))throw Error('Missing pipeline steps or comparison shaders');
  return {comparisons,frames:result.steps,particles:count,intermediateReadbackBytes:0,artifacts};
 }finally{resources.forEach(b=>runtime.destroyBuffer(b));textureResources.forEach(t=>runtime.destroyTexture(t));}
}
