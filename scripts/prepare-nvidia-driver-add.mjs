import {readFile,writeFile} from 'node:fs/promises';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';
const sample='cpp/0_Introduction/simpleDrvRuntime',file=sample+'/vectorAdd_kernel.cu',source=await readFile('.local/nvidia-audit/'+file,'utf8'),entry='VecAdd_kernel';
const rows=JSON.parse(await readFile('reports/nvidia-artifacts.json','utf8')).filter(r=>r.entry!==entry);
rows.push({sample,file,entry,wholeFile:true,artifact:serializableArtifact(compile(source,{entry,workgroupSize:[128,1,1]}))});
await writeFile('reports/nvidia-artifacts.json',JSON.stringify(rows));console.log('Compiled complete original driver/runtime vector-add kernel file.');
