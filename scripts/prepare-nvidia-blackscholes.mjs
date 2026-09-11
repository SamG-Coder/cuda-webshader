import {readFile,writeFile} from 'node:fs/promises';import {compile,serializableArtifact} from '../src/compiler/compiler.js';
const sample='cpp/5_Domain_Specific/BlackScholes',file=sample+'/BlackScholes_kernel.cuh',source=await readFile('.local/nvidia-audit/'+file,'utf8');
const rows=JSON.parse(await readFile('reports/nvidia-artifacts.json','utf8')).filter(r=>r.sample!==sample);
rows.push({sample,file,entry:'BlackScholesGPU',wholeFile:true,previewBufferOverrides:{d_StockPrice:{fill:'ramp',scale:0.5,offset:5},d_OptionStrike:{fill:'one',scale:50},d_OptionYears:{fill:'one'}},artifact:serializableArtifact(compile(source,{entry:'BlackScholesGPU',workgroupSize:[128,1,1]}))});
await writeFile('reports/nvidia-artifacts.json',JSON.stringify(rows));console.log('Compiled full original BlackScholes kernel file and both helper functions.');
