import {readFile,writeFile} from 'node:fs/promises';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';
import {kernelSource} from '../src/sandbox/import.js';
const sample='cpp/2_Concepts_and_Techniques/scalarProd',file=sample+'/scalarProd_kernel.cuh',source=await readFile('.local/nvidia-audit/'+file,'utf8');
const preamble=['#define IMUL(a, b) __mul24(a, b)','#define ACCUM_N 1024','namespace cg = cooperative_groups;'].map(line=>{if(!source.includes(line))throw Error('Upstream preamble changed');return line;}).join('\n')+'\n';
const artifact=serializableArtifact(compile(kernelSource(source).source,{entry:'scalarProdGPU',workgroupSize:[128,1,1]}));
const rows=JSON.parse(await readFile('reports/nvidia-artifacts.json','utf8')).filter(r=>r.sample!==sample);rows.push({sample,file,entry:'scalarProdGPU',preamble,artifact});await writeFile('reports/nvidia-artifacts.json',JSON.stringify(rows));console.log('Compiled unchanged NVIDIA scalar-product kernel.');
