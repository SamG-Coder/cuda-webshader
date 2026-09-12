import {readFileSync,writeFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';
let source='';for(const file of ['vec3.h','ray.h','hitable.h','material.h','sphere.h','hitable_list.h','camera.h']){
 let s=readFileSync('.local/raytracing-cuda/'+file,'utf8').replace(/^\s*#(?:include[^\n]*|ifndef[^\n]*|define (?:VEC3H|RAYH|HITABLEH|MATERIALH|SPHEREH|HITABLELISTH|CAMERAH)|endif)\s*$/gm,'').replace(/^struct hit_record;\s*$/gm,'').replace(/inline std::(?:istream|ostream)& operator[<>]{2}\([^)]*\)\s*\{[^}]*\}/g,'');source+=s+'\n';
}
const main=readFileSync('.local/raytracing-cuda/main.cu','utf8');source+='#define FLT_MAX 3.402823466e+38f\n'+main.slice(main.indexOf('__device__ vec3 color'),main.indexOf('int main()'));
writeFileSync('.local/full-pathtracer-device.cu',source);
const results=[];for(const entry of ['rand_init','render_init','create_world','render','free_world']){try{compile(source,{entry,valueBuffers:['fb'],libraries:['curand-xorwow'],objectHeap:'persistent',workgroupSize:[8,8,1]});results.push({entry,compiled:true});}catch(e){results.push({entry,compiled:false,error:e.message});}}

writeFileSync('reports/pathtracer-integration-probes.json',JSON.stringify(results,null,2)+'\n');console.log(JSON.stringify(results,null,2));
