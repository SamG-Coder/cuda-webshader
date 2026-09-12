import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {kernelSource} from '../src/sandbox/import.js';
import {compile} from '../src/compiler/compiler.js';
const source=await readFile('.local/nvidia-audit/cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu','utf8'),results=[];
try{const extracted=kernelSource(source);await writeFile('tests/quadtree-cdp-device.cuh',source.slice(0,source.indexOf('#include'))+extracted.source);compile(extracted.source+'\n'+await readFile('tests/quadtree-cdp-setup.cuh','utf8'),{entry:'build_quadtree_kernel<128>',valueBuffers:['nodes','points'],workgroupSize:[128],sharedMemoryBytes:64,objectHeap:'persistent',deviceHeap:{maxAllocations:1024,maxElements:1024},deviceLaunchQueue:{maxLaunches:1024}});results.push({compiled:true});}catch(e){results.push({compiled:false,error:e.message});}
const report={upstreamRevision:'5443602d89ed99aede2e4b7bf329daddeadb320e',sourceSha256:createHash('sha256').update(source).digest('hex'),originalDeviceBodies:true,results};await writeFile('reports/quadtree-cdp-probes.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
