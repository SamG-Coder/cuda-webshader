import {readdir,readFile,writeFile,mkdir} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import path from 'node:path';
const audit=JSON.parse(await readFile('reports/nvidia-audit.json','utf8'));
const knownArgs={simpleIPC:[],memMapIPCDrv:[],tileMatmul:[],tileMatmulAutotuner:[],ptxgen:[path.resolve('.local/nvidia-audit/cpp/7_libNVVM/ptxgen/test.ll')],UnifiedMemoryPerf:['--kernel-iterations=1'],Mandelbrot:['--file=Mandelbrot_fp32.ppm'],bicubicTexture:['--mode=2','--file=bicubic-output.ppm'],bilateralFilter:['--radius=5','--passes=1','--file=ref_05.ppm'],marchingCubes:['--file=Bucky.raw','--dump=0'],postProcessGL:['--radius=4','--file=teapot_4.ppm'],simpleCUDA2GL:['--file=ref_simpleCUDA2GL.ppm'],simpleTexture3D:['--file=ref_texture3D.bin'],FunctionPointers:['--mode=0','--file=ref_orig.pgm'],boxFilter:['--radius=14','--passes=1','--file=ref_14.ppm']};
await mkdir('.local/nvidia-checks',{recursive:true});await writeFile('.local/nvidia-checks/neutral.nv12',Buffer.alloc(32*32*3/2,128));
knownArgs.NV12toBGRandResize=['--input='+path.resolve('.local/nvidia-checks/neutral.nv12'),'--width=32','--height=32','--dst_width=16','--dst_height=16','--batch=1'];
let results=[];try{results=JSON.parse(await readFile('reports/nvidia-native.json','utf8')).results;}catch{}
for(const r of results)if(r.executable==='simpleAssert.exe'&&r.exitCode===0&&r.tail.includes('returned OK'))r.status='expected-assert-passed';
for(const r of results)if(r.status==='reported-failure'&&r.exitCode===0)r.status='exited-zero';
await mkdir('reports/nvidia-native-logs',{recursive:true});
for(const sample of audit.samples.filter(s=>s.language==='CUDA C++')){
 let files;const dir=path.resolve('.local/nvidia-audit-build',sample.id);try{files=await readdir(dir);}catch{continue;}
 for(const file of files.filter(f=>f.endsWith('.exe'))){const id=sample.id+'/'+file,stem=file.slice(0,-4),previous=results.find(r=>r.id===id);
 let args=knownArgs[stem]||(stem==='vectorAdd'?['1024']:['--qatest']);
 if(!knownArgs[stem]){let sources='';for(const name of await readdir(path.resolve('.local/nvidia-audit',sample.id))){if(/\.(cu|cpp)$/.test(name))sources+=await readFile(path.resolve('.local/nvidia-audit',sample.id,name),'utf8');}if(/checkCmdLineFlag\([^;]+"file"/.test(sources)){try{const refs=(await readdir(path.resolve('.local/nvidia-audit',sample.id,'data'))).filter(f=>/^ref[_\.]/i.test(f));if(refs.length)args=['--file='+refs[0]];}catch{}}}
 if(previous&&!(previous.exitCode===3221225781||previous.exitCode===-1073741515)&&JSON.stringify(previous.args)===JSON.stringify(args))continue;
 const timeoutMs=['UnifiedMemoryPerf','tileMatmulAutotuner'].includes(stem)?120000:12000;
 const start=Date.now();const result=await new Promise(resolve=>execFile(path.join(dir,file),args,{cwd:dir,timeout:timeoutMs,windowsHide:true,maxBuffer:4*1024*1024,env:{...process.env,PATH:path.resolve('.local/cuda-samples/bin/win64/Release')+';'+process.env.PATH}},(error,stdout,stderr)=>resolve({error,stdout,stderr})));
 const output=result.stdout+'\n'+result.stderr,code=result.error?.code??0,status=result.error?.killed?'timeout':code===2?'waived':code!==0?'execution-failed':'exited-zero';
 const log=`nvidia-native-logs/${sample.id.replaceAll('/','-')}-${stem}.txt`;await writeFile('reports/'+log,output);
 if(previous)results=results.filter(r=>r.id!==id);
 results.push({id,sample:sample.id,executable:file,args,timeoutMs,status:stem==='simpleAssert'&&code===0&&output.includes('returned OK')?'expected-assert-passed':status,exitCode:code,durationMs:Date.now()-start,log,tail:output.slice(-2000),...(previous?{previousAttempt:previous}: {})});
 await writeFile('reports/nvidia-native.json',JSON.stringify({date:new Date().toISOString(),method:'Upstream executables, --qatest where accepted; 12 second timeout. Exit zero alone is not proof of numerical correctness. Working directory is the sample build directory.',results},null,2));console.log(`${sample.name}: ${status}`);
 }
}
