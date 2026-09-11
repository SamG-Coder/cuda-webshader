import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {fixture} from '../showcases/nvidia/fixtures.js';
const rows=JSON.parse(await readFile('reports/nvidia-artifacts.json','utf8'));
await mkdir('showcases/nvidia/kernels',{recursive:true});await mkdir('.local/nvidia-checks',{recursive:true});
let native='#include <cuda_runtime.h>\n#include <cooperative_groups.h>\n#include <cstdio>\n#include <cmath>\nusing uint = unsigned int;\n#define CHECK(x) do {auto r=(x);if(r!=cudaSuccess){printf("CUDA ERROR %s\\n",cudaGetErrorString(r));return 2;}}while(0)\n';
const runs=[];
for(const [index,row] of rows.entries()){
 const text=await readFile('.local/nvidia-audit/'+row.file,'utf8'),start=text.search(new RegExp('__global__\\s+void\\s+'+(row.sourceEntry||row.entry)+'\\s*\\('));
 if(start<0)throw Error('Cannot locate '+row.entry);const brace=text.indexOf('{',start);let depth=1,end=brace+1;for(;depth;end++){if(text[end]==='{')depth++;if(text[end]==='}')depth--;if(end>=text.length)throw Error('Unbalanced kernel');}
 const source=row.wholeFile?text:text.slice(0,text.indexOf('*/')+2)+'\n\n'+(row.preamble||'')+text.slice(start,end)+'\n';
 row.source=`kernels/${index}.cu`;await writeFile('showcases/nvidia/'+row.source,source);
 native+=`namespace sample${index} {\n${row.wholeFile?text:(row.preamble||'')+text.slice(start,end)}\n}\n`;
 const f=fixture(row),meta=row.artifact.metadata,lines=[];
 row.preview={groups:f.groups,scalars:f.scalars,output:f.out,buffers:Object.fromEntries(meta.bindings.map(b=>[b.name,{records:f.buffers[b.name].length*4/b.stride,fill:b.readOnly?'ramp':'zero',...row.previewBufferOverrides?.[b.name]}]))};
 for(const b of meta.bindings){const a=f.buffers[b.name],type=a instanceof Uint32Array?'unsigned int':a instanceof Int32Array?'int':'float',name=b.name;lines.push(`${type} h_${name}[]={${Array.from(a,v=>v===-2147483648?'(-2147483647 - 1)':String(v)).join(',')}}; ${type} *d_${name}; CHECK(cudaMalloc(&d_${name},sizeof(h_${name}))); CHECK(cudaMemcpy(d_${name},h_${name},sizeof(h_${name}),cudaMemcpyHostToDevice));`);}
 // Parameter order follows the unchanged CUDA declaration, rather than grouped metadata.
 const signature=text.slice(start,brace),params=signature.slice(signature.indexOf('(')+1,signature.lastIndexOf(')')).split(',').map(p=>p.trim().match(/(\w+)\s*$/)[1]);
 const args=params.map(p=>p in f.buffers?`(${meta.bindings.find(b=>b.name===p).elementType.startsWith('vec')?(f.buffers[p] instanceof Uint32Array?'uint':f.buffers[p] instanceof Int32Array?'int':'float')+meta.bindings.find(b=>b.name===p).stride/4: f.buffers[p] instanceof Uint32Array?'unsigned int':f.buffers[p] instanceof Int32Array?'int':'float'}*)d_${p}`:String(f.scalars[p]));
 lines.push(`sample${index}::${row.entry}<<<dim3(${f.groups}),dim3(${meta.workgroupSize})>>>(${args}); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); int failures=0;`);
 for(const [name,expected]of Object.entries(f.expectedOutputs||{[f.out]:f.expected}))lines.push(`{CHECK(cudaMemcpy(h_${name},d_${name},sizeof(h_${name}),cudaMemcpyDeviceToHost)); double expected[]={${Array.from(expected,v=>Number(v).toExponential(17)).join(',')}};for(int i=0;i<${expected.length};i++)if(!std::isfinite(h_${name}[i])||fabs(h_${name}[i]-expected[i])>${f.absoluteTolerance??0.000003}+${f.relativeTolerance??0}*fabs(expected[i]))failures++;}`);
 lines.push(`printf("${index} %s\\n",failures?"FAIL":"PASS"); total+=failures;`);
 for(const b of meta.bindings)lines.push(`CHECK(cudaFree(d_${b.name}));`);
 runs.push('{'+lines.join('\n')+'}');
}
native+='int main(){int total=0;'+runs.join('\n')+'return total?1:0;}';
await writeFile('.local/nvidia-checks/check.cu',native);await writeFile('showcases/nvidia/artifacts.json',JSON.stringify(rows));
console.log(`Prepared ${rows.length} unchanged kernel fixtures.`);
