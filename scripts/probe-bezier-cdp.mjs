import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {compile} from '../src/compiler/compiler.js';
import {kernelSource} from '../src/sandbox/import.js';
const path='.local/nvidia-audit/cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu';
const source=await readFile(path,'utf8'), results=[];
for(const entry of ['computeBezierLinePositions','computeBezierLinesCDP','freeVertexMem']){
 try{const extracted=kernelSource(source);compile(extracted.source,{entry,workgroupSize:[32,1,1]});results.push({entry,compiled:true});}
 catch(error){results.push({entry,compiled:false,error:error.message});}
}
const report={upstreamRevision:'5443602d89ed99aede2e4b7bf329daddeadb320e',sourceSha256:createHash('sha256').update(source).digest('hex'),originalDeviceBodies:true,results};
await writeFile('reports/bezier-cdp-probes.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
