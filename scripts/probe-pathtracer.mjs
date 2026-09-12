import {readFile,writeFile} from 'node:fs/promises';
import {compile} from '../src/compiler/compiler.js';
const root=new URL('../.local/raytracing-cuda/',import.meta.url),results=[];
for(const file of ['main.cu','vec3.h','vec3-device-only','ray.h','hitable.h','sphere.h']) {
 const source=await readFile(new URL(file==='vec3-device-only'?'vec3.h':file,root),'utf8');
 // Header probes retain original declarations and bodies. Removing includes
 // isolates language support from external library/header resolution.
 let candidate=file==='main.cu'?source:source.replace(/^\s*#include[^\n]*$/gm,'').replace(/^\s*#(?:ifndef\s+(?:VEC3H|HITABLEH|RAYH|SPHEREH)|define\s+(?:VEC3H|HITABLEH|RAYH|SPHEREH)|endif)\s*$/gm,'')+'\n__global__ void probe(float*out){out[0]=0.0f;}';
 // Explicitly isolate device declarations; never rewrite CUDA method bodies.
 if(file==='vec3-device-only')candidate=candidate.replace(/inline std::(?:istream|ostream)& operator[<>]{2}\([^)]*\)\s*\{[^}]*\}/g,'');
 if(['ray.h','hitable.h','sphere.h'].includes(file)){
   const dependencies=['vec3.h',...(file!=='ray.h'?['ray.h']:[]),...(file==='sphere.h'?['hitable.h']:[])];let prelude='';
   for(const dependency of dependencies){const text=await readFile(new URL(dependency,root),'utf8');prelude+=text.replace(/^\s*#.*$/gm,'').replace(/inline std::(?:istream|ostream)& operator[<>]{2}\([^)]*\)\s*\{[^}]*\}/g,'')+'\n';}
   candidate=prelude+candidate;
 }
 try{compile(candidate,{entry:file==='main.cu'?'render':'probe',workgroupSize:[8,8,1]});results.push({file,compiled:true});}
 catch(error){results.push({file,compiled:false,error:error.message});}
}
await writeFile(new URL('../reports/pathtracer-compiler-probes.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
console.log(JSON.stringify(results,null,2));
