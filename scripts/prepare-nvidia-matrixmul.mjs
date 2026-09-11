import {readFile,writeFile} from 'node:fs/promises';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';
const sample='cpp/0_Introduction/matrixMul',file=sample+'/matrixMul.cu',text=await readFile('.local/nvidia-audit/'+file,'utf8'),start=text.indexOf('template <int BLOCK_SIZE> __global__ void MatrixMulCUDA'),brace=text.indexOf('{',start);let end=brace+1,depth=1;for(;depth;end++){if(text[end]==='{')depth++;if(text[end]==='}')depth--;if(end>=text.length)throw Error('Unbalanced kernel');}
if(start<0)throw Error('Original template declaration missing');
const source=text.slice(0,text.indexOf('*/')+2)+'\n\n'+text.slice(start,end)+'\n',rows=JSON.parse(await readFile('reports/nvidia-artifacts.json','utf8')).filter(r=>r.sample!==sample);
for(const tile of [16,32]){const entry=`MatrixMulCUDA<${tile}>`;rows.push({sample,file,entry,sourceEntry:'MatrixMulCUDA',preamble:'template <int BLOCK_SIZE> ',artifact:serializableArtifact(compile(source,{entry,workgroupSize:[tile,tile,1]}))});}
await writeFile('reports/nvidia-artifacts.json',JSON.stringify(rows));console.log('Compiled original matrix multiplication kernel at tiles 16 and 32.');
