import {readFile,writeFile,mkdir,access} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import path from 'node:path';
const root=path.resolve('.local/nvidia-audit-build'),dirs=(await readFile(root+'/CMakeFiles/TargetDirectories.txt','utf8')).split(/\r?\n/),results=[];
await mkdir('reports/nvidia-build-logs',{recursive:true});
for(const dir of dirs){const match=dir.replaceAll('\\','/').match(/\/(cpp\/.+)\/CMakeFiles\/([^/]+)\.dir$/);if(!match||['edit_cache','rebuild_cache','list_install_components','install'].includes(match[2]))continue;const [,sample,target]=match;let exists=false;try{await access(root+'/'+sample+'/'+target+'.exe');exists=true;}catch{}
if(exists){results.push({sample,target,status:'built',evidence:'Executable produced by root build'});continue;}
const result=await new Promise(resolve=>execFile('cmake',['--build',root,'--target',target],{timeout:180000,windowsHide:true,maxBuffer:8*1024*1024},(error,stdout,stderr)=>resolve({error,output:stdout+'\n'+stderr})));
const log='nvidia-build-logs/'+sample.replaceAll('/','-')+'-'+target+'.txt';await writeFile('reports/'+log,result.output);const status=result.error?.killed?'build-timeout':result.error?'build-failed':'built';results.push({sample,target,status,log,tail:result.output.slice(-7000)});await writeFile('reports/nvidia-build-targets.json',JSON.stringify({results},null,2));console.log(target+': '+status);
}
await writeFile('reports/nvidia-build-targets.json',JSON.stringify({results},null,2));
