import {readdir,readFile,writeFile,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {kernelSource} from '../src/sandbox/import.js';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';
const root=path.resolve(process.env.CW_NVIDIA_ROOT||'.local/nvidia-audit'),commit=execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
async function walk(dir){const results=[];for(const e of await readdir(dir,{withFileTypes:true})){const p=path.join(dir,e.name);if(e.isDirectory())results.push(...await walk(p));else results.push(p);}return results;}
const files=[...await walk(path.join(root,'cpp')),...await walk(path.join(root,'python'))],directories=new Set(files.filter(f=>path.basename(f)==='README.md').map(f=>path.dirname(f)).filter(f=>path.relative(root,f).split(path.sep).length>=3&&!f.endsWith(`${path.sep}Tegra`)));
const artifacts=[],samples=[];
const patterns=[['C++ templates',/\btemplate\s*</],['texture/surface API',/\b(?:tex[123]D|surf\w+|cudaTextureObject_t|cudaSurfaceObject_t)\b/],['warp intrinsics / inline PTX',/\b(?:__shfl\w*|__ballot\w*|asm|__syncwarp)\b/],['cooperative groups',/cooperative_groups|\bcg::/],['double / half / tensor types',/\b(?:double|half|__half|__nv_bfloat16|wmma|mma|__nv_fp8\w*)\b/],['constant / dynamic shared memory',/__constant__|extern\s+__shared__/],['external CUDA library',/\b(?:cublas\w*|cufft\w*|cusparse\w*|curand\w*|cub::|thrust::)/],['multiple GPUs / peer access',/cudaDeviceCanAccessPeer|cudaDeviceEnablePeerAccess|MPI_/],['OS / external graphics interop',/cudaGraphics|NvSci|vulkan|D3D|EGLStream/]];
for(const dir of [...directories].sort()){
 const relative=path.relative(root,dir).replaceAll(path.sep,'/'),sampleFiles=files.filter(f=>path.dirname(f)===dir),readme=await readFile(path.join(dir,'README.md'),'utf8');
 const sample={id:relative,name:path.basename(dir),category:relative.split('/')[1],language:relative.startsWith('python/')?'Python':'CUDA C++',url:`https://github.com/NVIDIA/cuda-samples/tree/${commit}/${relative}`,summary:readme.replace(/\r/g,'').split('\n').filter(l=>l.trim()&&!l.startsWith('#'))[0]||'',features:[],probes:[],native:{status:'not-run'},browser:{status:'not-run'}};
 let combined='';
 for(const file of sampleFiles.filter(f=>/\.(cu|cuh|cpp|h)$/.test(f))){const text=await readFile(file,'utf8');combined+=text+'\n';if(!text.includes('__global__'))continue;const entries=[...text.matchAll(/__global__\s+(?:__launch_bounds__\([^)]*\)\s*)?void\s+(\w+)\s*\(/g)].map(m=>m[1]);
  for(const entry of [...new Set(entries)]){
   const probe={file:path.relative(root,file).replaceAll(path.sep,'/'),entry};
   try{const input=kernelSource(text);const block=/threadIdx\.y/.test(input.source)?[8,8,1]:[128,1,1];const artifact=serializableArtifact(compile(input.source,{entry,workgroupSize:block}));probe.translation='passed';probe.block=block;probe.artifactIndex=artifacts.length;artifacts.push({sample:relative,file:probe.file,entry,artifact});}
   catch(e){probe.translation='rejected';probe.error=e.message.slice(0,700);}
   sample.probes.push(probe);
  }
 }
 sample.features=patterns.filter(([,p])=>p.test(combined)).map(([name])=>name);
 if(sample.language==='Python')sample.browser={status:'unsupported-input',reason:'Python/NumPy/CuPy/framework sample; this translator accepts CUDA C kernels, not Python programs.'};
 else if(sample.probes.some(p=>p.translation==='passed'))sample.browser={status:'translation-passed',reason:'One or more kernels translated. GPU compilation and execution are separate checks.'};
 else if(sample.probes.length)sample.browser={status:'translation-rejected',reason:sample.probes[0].error};
 else sample.browser={status:'no-extractable-kernel',reason:'No standalone __global__ void kernel extracted from direct sample source files. May use templates, libraries, generated code or host-only utilities.'};
 if(relative.includes('/Tegra/'))sample.native={status:'platform-blocked',reason:'Tegra/Jetson-specific APIs and hardware are unavailable on this Windows RTX 5080 desktop.'};
 if(sample.language==='Python')sample.native={status:'not-run',reason:'Python environment and framework dependencies must be evaluated separately from NVCC samples.'};
 samples.push(sample);
}
await mkdir('reports',{recursive:true});await writeFile('reports/nvidia-audit.json',JSON.stringify({commit,date:new Date().toISOString(),scope:'All sample README directories in cpp/ and python/ at the pinned upstream commit',method:'Actual local CUDA-to-WGSL compiler probes; source feature findings are not execution results.',samples},null,2));await writeFile('reports/nvidia-artifacts.json',JSON.stringify(artifacts));
console.log(JSON.stringify({samples:samples.length,kernelProbes:samples.reduce((n,s)=>n+s.probes.length,0),translated:artifacts.length,byStatus:Object.fromEntries([...new Set(samples.map(s=>s.browser.status))].map(k=>[k,samples.filter(s=>s.browser.status===k).length]))},null,2));
