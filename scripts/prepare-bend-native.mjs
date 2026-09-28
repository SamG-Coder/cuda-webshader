import {readFile,writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {compileBend} from '../src/bend/compiler.js';
import {packBendInputs} from '../src/bend/runtime.js';
import {bendCases} from '../tests/bend-cases.js';
const directory=process.argv[2];
if(!directory)throw Error('Usage: node scripts/prepare-bend-native.mjs <output directory>');
await mkdir(directory,{recursive:true});
const runtime=await readFile(new URL('../src/bend/runtime.cu',import.meta.url),'utf8');
let source='#include <cuda_runtime.h>\n#include <cstdio>\n#include <cstdint>\n#include <cstdlib>\n\n';
const runs=[];
for(const [i,sample] of bendCases.entries()) {
  const c=compileBend(sample.source,runtime,{entry:sample.entry});
  const input=packBendInputs(c.program,sample.rows);
  const expected=c.program.resultType==='F32'?new Uint32Array(new Float32Array(sample.expected).buffer):new Uint32Array(sample.expected);
  source+=`namespace case_${i} {\n${c.cuda}\n#undef BEND_ROOT\n#undef BEND_ARGS\n#undef BEND_RESULT\n}\n`;
  runs.push(`{
    const unsigned int count=${sample.rows.length}u;
    unsigned int input[]={${[...input].map(x=>x+'u').join(',')}};
    unsigned int expected[]={${[...expected].map(x=>x+'u').join(',')}};
    unsigned int *di,*out,*status,*usage,*heap;
    CHECK(cudaMalloc(&di,sizeof(input))); CHECK(cudaMalloc(&out,count*4));
    CHECK(cudaMalloc(&status,count*4));CHECK(cudaMalloc(&usage,count*8));
    CHECK(cudaMalloc(&heap,count*65536u*4u));
    CHECK(cudaMemcpy(di,input,sizeof(input),cudaMemcpyHostToDevice));
    case_${i}::bend_run<<<(count+63)/64,64>>>(di,out,status,usage,heap,count,65536u,100000u);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    unsigned int actual[${sample.rows.length}],errors[${sample.rows.length}];
    CHECK(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(errors,status,sizeof(errors),cudaMemcpyDeviceToHost));
    for(unsigned int j=0;j<count;j++) if(errors[j] || actual[j]!=expected[j]) {
      std::fprintf(stderr,"${sample.name} row %u: status=%u got=%u expected=%u\\n",j,errors[j],actual[j],expected[j]);return 1;
    }
    CHECK(cudaFree(di));CHECK(cudaFree(out));CHECK(cudaFree(status));CHECK(cudaFree(usage));CHECK(cudaFree(heap));
    std::printf("PASS ${sample.name}: %u results\\n",count);
  }`);
}
source+='#define CHECK(expr) do { cudaError_t err=(expr); if(err!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\\n",cudaGetErrorString(err));std::exit(2);} } while(0)\n';
source+='int main(){\n'+runs.join('\n')+'\nstd::puts("ALL BEND NATIVE CASES PASSED");return 0;\n}\n';
await writeFile(path.join(directory,'bend-native.cu'),source);
console.log(`Prepared ${bendCases.length} native cases in ${directory}`);
