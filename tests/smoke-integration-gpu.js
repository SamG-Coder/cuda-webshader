export async function checkSmokeIntegration(runtime){
 const source=await(await fetch('/tests/smoke-integration-kernel.cuh')).text();
 const integrate=await runtime.kernel(source,{entry:'integrate_functor',workgroupSize:[128,1,1]}),depth=await runtime.kernel(source,{entry:'calcDepth_functor',workgroupSize:[128,1,1]});
 const load=async name=>new Float32Array(await(await fetch('/reports/'+name+'.bin')).arrayBuffer());
 const initialP=await load('smoke-integration-input-positions'),initialV=await load('smoke-integration-input-velocities'),count=initialP.length/4,resources=[];
 const buffer=(values,length)=>{const data=new Float32Array(length+16).fill(-77);data.set(values);const b=runtime.createBuffer(data);resources.push(b);return b;};
 let a=buffer(initialP,count*4),b=buffer(initialV,count*4),c=buffer(new Float32Array(count*4),count*4),d=buffer(new Float32Array(count*4),count*4);const keys=buffer(new Float32Array(count),count);
 const texture=runtime.createTexture3D(await load('smoke-noise-input'),{width:64,height:64,depth:64,format:'rgba32float',filter:'linear',addressMode:'repeat'});
 const constants=Object.fromEntries(Object.entries({'gravity.x':0,'gravity.y':.0001,'gravity.z':0,globalDamping:.99,noiseFreq:.1,noiseAmp:.001,'noiseSpeed.x':.01,'noiseSpeed.y':.02,'noiseSpeed.z':-.01}).map(([k,v])=>['constant.cudaParams.'+k,v])),comparisons=[];
 try{
  for(let frame=1;frame<=64;frame++){
   runtime.batch().dispatch(integrate.bind({tuple0:c,tuple1:d,tuple2:a,tuple3:b,noiseTex:texture},{...constants,deltaTime:.5,count,'constant.cudaParams.time':(frame-1)*.5}),[Math.ceil(count/128),1,1]).submit();
   [a,c]=[c,a];[b,d]=[d,b];
   if([1,8,32,64].includes(frame)){
    const direction=frame<32?[.25,-.5,1]:[-1,.125,.5];
    runtime.batch().dispatch(depth.bind({tuple0:a,tuple1:keys},{count,...Object.fromEntries(direction.map((v,i)=>['sortVector.'+'xyz'[i],v]))}),[Math.ceil(count/128),1,1]).submit();
    for(const [name,resource,size] of [['positions',a,count*4],['velocities',b,count*4],['depth',keys,count]]){
     const actual=await runtime.read(resource),expected=await load('smoke-integration-'+frame+'-'+name);if(expected.length!==size)throw Error('Invalid native capture length');
     let maxError=0;for(let i=0;i<size;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite smoke output');maxError=Math.max(maxError,Math.abs(actual[i]-expected[i]));}
     for(let i=size;i<actual.length;i++)if(actual[i]!==-77)throw Error('Smoke partial block overwrote guard');
     if(maxError>2e-6)throw Error('Native smoke mismatch '+name+' frame '+frame+': '+maxError);
     comparisons.push({frame,name,components:size,maxError});
    }
   }
  }
  return {count,frames:64,comparisons,originalFunctorsUnchanged:true,partialBlockGuards:true};
 }finally{resources.forEach(b=>runtime.destroyBuffer(b));runtime.destroyTexture(texture);}
}
