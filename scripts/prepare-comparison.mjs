import {mkdir,writeFile} from 'node:fs/promises';
import {makeComparisonCases} from '../benchmarks/cases.js';
const root=new URL('../',import.meta.url),out=new URL('reports/comparison-inputs/',root);await mkdir(out,{recursive:true});
let code='#include "../../benchmarks/native-support.cuh"\nint main(){try{std::cout<<std::setprecision(10);cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));std::cerr<<"GPU: "<<prop.name<<" SM "<<prop.major<<"."<<prop.minor<<"\\n";\n';
const manifest=[];
for(const scale of ['small','large'])for(const c of makeComparisonCases(scale)){
 const prefix=`reports/comparison-inputs/${c.key}`,names=Object.keys(c.buffers);code+='{\n';
 for(const [name,array] of Object.entries(c.buffers)){await writeFile(new URL(`${c.key}-${name}.bin`,out),new Uint8Array(array.buffer));code+=`Buffer ${name}("${prefix}-${name}.bin");\n`;}
 for(const [name,array] of Object.entries(c.expected))await writeFile(new URL(`${c.key}-${name}-expected.bin`,out),new Uint8Array(array.buffer));
 let launch='';
 if(c.id==='reduce_sum'){let n=c.scalars.n,i=0,prev='input';do{const count=Math.ceil(n/256),next=`level${i++}`;code+=`Buffer ${next}(${count*4});\n`;launch+=`reduce_sum<<<${count},128,0,stream>>>((float*)${prev}.p,(float*)${next}.p,${n}u);`;prev=next;n=count;}while(n>1);code+=`Buffer& output=${prev};\n`;}
 else {const pointers=names.map(name=>`(${c.buffers[name] instanceof Uint32Array?'unsigned int':c.id==='saxpy_vec4'||c.id==='particles'?'float4':'float'}*)${name}.p`),scalars=Object.entries(c.scalars).map(([k,v])=>['a','dt','time','attraction'].includes(k)?`${Number(v).toFixed(10)}f`:`${v}u`);if(c.id==='histogram')launch+='clear_bins<<<1,256,0,stream>>>((unsigned int*)bins.p);';launch+=`${c.id}<<<dim3(${c.groups.join(',')}),dim3(${c.block.join(',')}),0,stream>>>(${[...pointers,...scalars].join(',')});`;}
 code+=`auto launch=[&](cudaStream_t stream){${launch}ck(cudaGetLastError());};\nlaunch(nullptr);ck(cudaDeviceSynchronize());\n`;
 for(const [name,array] of Object.entries(c.expected))code+=`${name}.verify("${prefix}-${name}-expected.bin",${array instanceof Uint32Array});\n`;
 code+=`benchmark("${c.key}",${c.iterations},launch,[&](){${names.map(n=>`${n}.reset();`).join('')}});\n}\n`;
 const {buffers,expected,...metadata}=c;manifest.push({...metadata,buffers:Object.fromEntries(Object.entries(buffers).map(([k,v])=>[k,{type:v.constructor.name,bytes:v.byteLength}]))});
}
code+='return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}\n';
await writeFile(new URL('native-generated.cu',out),code);await writeFile(new URL('../comparison-manifest.json',out),JSON.stringify(manifest,null,2));console.log('Prepared 20 matched cases and native CUDA launcher.');
