import {readFile,writeFile} from 'node:fs/promises';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';
const file='cpp/6_Performance/transpose/transpose.cu',source=await readFile('.local/nvidia-audit/'+file,'utf8');
const preamble=['#define TILE_DIM   32','#define BLOCK_ROWS 16','namespace cg = cooperative_groups;'].map(line=>{if(!source.includes(line))throw Error('Upstream preamble changed: '+line);return line;}).join('\n')+'\n';
const rows=JSON.parse(await readFile('reports/nvidia-artifacts.json','utf8')).filter(r=>r.entry!=='transposeNaive');
for(const entry of ['transposeNaive']){
 const start=source.indexOf('__global__ void '+entry+'('),brace=source.indexOf('{',start);let end=brace+1,depth=1;for(;depth;end++){if(source[end]==='{')depth++;if(source[end]==='}')depth--;if(end>=source.length)throw Error('Unbalanced source');}
 const artifact=serializableArtifact(compile(preamble+source.slice(start,end),{entry,workgroupSize:[32,16,1]}));
 rows.push({sample:'cpp/6_Performance/transpose',file,entry,preamble,artifact});
}
await writeFile('reports/nvidia-artifacts.json',JSON.stringify(rows));console.log('Compiled original NVIDIA naive transpose kernel.');
