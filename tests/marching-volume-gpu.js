import {executePipeline} from '../src/sandbox/pipeline.js';
import {compile} from '../src/compiler/compiler.js';
export async function checkMarchingVolume(runtime){
 const source=await(await fetch('/showcases/marching-cubes/kernel.cu')).text(),config=await(await fetch('/showcases/marching-cubes/volume-pipeline.json')).json(),compiler={compile:async(source,options)=>({artifact:compile(source,options)})},results=[];
 for(const [scenario,iso]of [.2,.5,.8].entries()){
  const plan=structuredClone(config.pipeline);for(const t of Object.values(plan.textures))if(t.source)t.source="/"+t.source;for(const step of plan.steps)if(step.scalars&&'isoValue'in step.scalars)step.scalars.isoValue=iso;
  const resources=[],textureResources=[],before=runtime.stats.readbackBytes;
  try{const result=await executePipeline({plan,source,defines:config.defines,compiler,runtime,rootArtifact:compile(source,{entry:'classifyVoxel',workgroupSize:[128,1,1],defines:config.defines}),resources,textureResources});
   if(runtime.stats.readbackBytes-before!==8)throw Error('Unexpected intermediate readback');const expectedVertices=[43524,33378,11214][scenario],active=[7164,5545,2288][scenario];if(result.count!==expectedVertices)throw Error('Wrong mesh count');
   const comparisons=[];for(const [name,key,Type,count]of [['positions','pos',Float32Array,result.count*4],['normals','norm',Float32Array,result.count*4],['scan','vertexScan',Uint32Array,32768],['compacted','compacted',Uint32Array,active]]){const actual=await runtime.read(result.buffers[key],Type,count*4),native=new Type(await(await fetch('/reports/marching-volume-native-'+scenario+'-'+name+'.bin')).arrayBuffer());if(actual.length!==native.length)throw Error('Native capture size mismatch');let maxError=0;for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite '+name);maxError=Math.max(maxError,Math.abs(actual[i]-native[i]));}if(maxError>(Type===Uint32Array?0:1e-6))throw Error('Bucky '+iso+' '+name+' mismatch '+maxError);comparisons.push({name,components:count,maxError});}
   results.push({iso,activeVoxels:active,vertices:result.count,triangles:result.count/3,controlReadbackBytes:8,comparisons});
  }finally{for(const r of resources)runtime.destroyBuffer(r);for(const t of textureResources)runtime.destroyTexture(t);}
 }return {originalSharedKernel:true,originalBuckyVolume:true,nativeFixturesUsedOnlyAsExpectedResults:true,results};
}
