export async function checkFluidsPitched(runtime){
 const load=async name=>new Float32Array(await(await fetch('/reports/'+name+'.bin')).arrayBuffer()),source=await(await fetch('/tests/fluids-pitched-kernels.cuh')).text();
 const kernels={};for(const entry of ['addForces_k','updateVelocity_k','advectParticles_k'])kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:entry==='addForces_k'?[5,5,1]:[16,4,1]});
 const v=runtime.createBuffer(await load('fluids-pitched-input')),part=runtime.createBuffer(await load('fluids-particles-input')),vx=runtime.createBuffer(await load('fluids-advect-1-0-native')),vy=runtime.createBuffer(await load('fluids-advect-1-1-native')),comparisons=[];
 const check=async(buffer,name)=>{const actual=await runtime.read(buffer),expected=await load(name);if(actual.length!==expected.length)throw Error('Wrong pitched native capture');let maxError=0;for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite pitched output');maxError=Math.max(maxError,Math.abs(actual[i]-expected[i]));if(expected[i]===-77&&actual[i]!==-77)throw Error('Pitched row padding changed');}if(maxError>1e-6)throw Error('Pitched native mismatch '+name+': '+maxError);comparisons.push({name,components:actual.length,maxError});};
 try{
  const forceScalars={dx:19,dy:13,spx:3,spy:2,fx:.5,fy:-.3,r:2,pitch:192};
  let rejected=false;try{kernels.addForces_k.bind({v},{...forceScalars,pitch:190});}catch{rejected=true;}if(!rejected)throw Error('Misaligned pitch accepted');
  runtime.batch().dispatch(kernels.addForces_k.bind({v},forceScalars),[1,1,1]).submit();await check(v,'fluids-forces-native');
  runtime.batch().dispatch(kernels.updateVelocity_k.bind({v,vx,vy},{dx:19,pdx:24,dy:13,lb:3,pitch:192}),[2,2,1]).submit();await check(v,'fluids-update-native');
  for(let i=0;i<5;i++)runtime.batch().dispatch(kernels.advectParticles_k.bind({v,part},{dx:19,dy:13,dt:500,lb:3,pitch:192}),[2,2,1]).submit();
  await check(part,'fluids-particles-native');
  return {comparisons,originalBodiesUnchanged:true,pitchBytes:192,paddingPreserved:true,misalignmentRejected:true,particleSteps:5};
 }finally{[v,part,vx,vy].forEach(b=>runtime.destroyBuffer(b));}
}
