import {GpuRuntime} from '../src/runtime/runtime.js';import {loadKernelSources} from '../src/kernels.js';import {runGpuSuite} from './gpu-suite.js';import {runThreeInteropTest} from './three-interop.js';import {runBenchmarks} from '../src/runtime/benchmark.js';
const output=document.getElementById('status');let runtime;
try{
 runtime=await GpuRuntime.create();const sources=await loadKernelSources();const report=await runGpuSuite(runtime,sources,{onCase:(r,n)=>{output.textContent=`${n}: ${r.pass?'PASS':'FAIL'} ${r.name}`;}});
 let interop;try{interop=await runThreeInteropTest(runtime);}catch(error){interop={name:'Three.js rendered-pixel interop',pass:false,error:String(error.stack||error)};}
 report.results.push(interop);report.total++;interop.pass?report.passed++:report.failed++;
 if(new URLSearchParams(location.search).has('bench')&&!report.failed)report.benchmarks=await runBenchmarks(runtime,sources,{n:262144,matrixSize:128,samples:3});
 window.__gpuReport=report;output.textContent=JSON.stringify(report,null,2);
}catch(error){window.__gpuReport={mode:'NOT RUN / initialization failure',failed:1,error:String(error.stack||error)};output.textContent=JSON.stringify(window.__gpuReport,null,2);}finally{runtime?.dispose();}
