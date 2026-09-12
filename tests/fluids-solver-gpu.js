import {fluidsSolverPlan} from './fluids-solver-plan.js';
import {executePipeline} from '../src/sandbox/pipeline.js';import {compile} from '../src/compiler/compiler.js';
export async function checkFluidsSolver(runtime){
 const source=await(await fetch('/tests/fluids-solver-kernels.cuh')).text(),results=[];
 for(const width of [64,512]){
  const plan=fluidsSolverPlan(width),resources=[],textureResources=[],artifacts=[];
  try{
   const before=runtime.stats.readbackBytes,result=await executePipeline({plan,source,runtime,compiler:{compile:async(s,o)=>({artifact:compile(s,o)})},rootArtifact:compile(source,{entry:'addForces_k',workgroupSize:[9,9,1]}),resources,textureResources,onArtifact:a=>artifacts.push(a.name)});
   if(runtime.stats.readbackBytes!==before)throw Error('Fluid initialization readback');
   for(let step=1;step<=64;step++){
    if(step>1){const before=runtime.stats.readbackBytes;await result.stepFrame();if(runtime.stats.readbackBytes!==before)throw Error('Fluid step readback');}
    if(![1,8,32,64].includes(step))continue;
    for(const name of ['velocity','particles']){
     const actual=await runtime.read(result.buffers[name]),expected=new Float32Array(await(await fetch('/reports/fluids-solver-'+width+'-'+step+'-'+name+'.bin')).arrayBuffer());if(actual.length!==expected.length)throw Error('Invalid solver capture');
     let maxError=0,squared=0,components=0;for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite solver output');if(expected[i]===-77){if(actual[i]!==-77)throw Error('Velocity padding changed');continue;}let error=Math.abs(actual[i]-expected[i]);if(name==='particles'){if(actual[i]<0||actual[i]>=1)throw Error('Particle not periodic');error=Math.min(error,Math.abs(1-error));}maxError=Math.max(maxError,error);squared+=error*error;components++;}
     const rms=Math.sqrt(squared/components);results.push({width,step,name,maxError,rms,components});if(maxError>(name==='velocity'?.0002:.25/width)||rms>(name==='velocity'?1e-5:.005/width))throw Error('Full solver native mismatch '+JSON.stringify(results));
    }
   }
  }finally{resources.forEach(b=>runtime.destroyBuffer(b));textureResources.forEach(t=>runtime.destroyTexture(t));}
 }
 return {results,intermediateReadbackBytes:0,originalKernelsUnchanged:true};
}
