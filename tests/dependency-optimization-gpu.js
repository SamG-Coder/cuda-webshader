import {GpuRuntime} from '../src/runtime/runtime.js';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {optimizerCases} from './dependency-optimization-cases.js';
export async function checkDependencyOptimization(runtime) {
 const results=[];let wordsCompared=0;
 for(const fixture of optimizerCases){
  const artifacts=[false,'dependencies','specialize'].map(optimize=>compile(fixture.source,{optimize,workgroupSize:[32]}));
  const kernels=[];for(const artifact of artifacts)kernels.push(await runtime.kernel(artifact));
  const output=runtime.createBuffer(128);
  try{
   for(const input of [-3,0,1,2,7,19]){
    const expected=new Int32Array(32);executeCPU(artifacts[0],{output:expected},{input},[1]);
    for(const k of kernels){
     runtime.batch().clear(output).dispatch(k.bind({output},{input}),[1]).submit();
     const actual=await runtime.read(output,Int32Array);
     for(let i=0;i<32;i++)if(actual[i]!==expected[i])throw Error(`${fixture.name} input=${input} word=${i}: ${actual[i]} !== ${expected[i]}`);
     wordsCompared+=32;
    }
   }
   results.push({case:fixture.name,exact:true});
  }finally{runtime.destroyBuffer(output);}
 }
 return {cases:results,wordsCompared,variants:['none','dependencies','specialize']};
}
export async function runOptimizerGpu(){
 const runtime=await GpuRuntime.create();
 try{return {adapter:runtime.describe(),...await checkDependencyOptimization(runtime)};}
 finally{runtime.dispose();}
}
