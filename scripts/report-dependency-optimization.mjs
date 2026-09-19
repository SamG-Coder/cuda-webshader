import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,dirname,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {compile,COMPILER_VERSION} from '../src/compiler/compiler.js';
// No GPU timings: this reports generated WGSL bytes and CPU translation only.
const root=resolve(process.argv[2]||'.integration/stratum');
const destination=resolve(process.argv[3]||'reports/dependency-optimization-stratum.json');
const shaderDir=process.argv[4]?resolve(process.argv[4]):null;
const {RENDER_SPECS}=await import(pathToFileURL(join(root,'src/kernel-specs.js')).href);
const sha=s=>createHash('sha256').update(s).digest('hex');
const results=[];
for(const spec of RENDER_SPECS){
 const parts=await Promise.all(spec.dependencies.map(n=>readFile(join(root,'kernels',n+'.cu'),'utf8')));
 const source=parts.join('\n'),variants=[];let originalInterface;
 for(const optimize of [false,'dependencies','specialize']){
  const start=performance.now(),c=compile(source,{entry:spec.entry,workgroupSize:spec.workgroupSize,optimize});
  const elapsed=performance.now()-start;
  const {optimization,...abi}=c.metadata;const signature=JSON.stringify(abi);
  if(originalInterface!==undefined&&signature!==originalInterface)throw Error(spec.entry+': optimizer changed interface');
  originalInterface??=signature;
  variants.push({mode:optimize===false?'none':optimize,bytes:Buffer.byteLength(c.wgsl),sha256:sha(c.wgsl),
   cpuCompileMs:elapsed,interfaceSha256:sha(signature),optimization});
  if(shaderDir){await mkdir(shaderDir,{recursive:true});await writeFile(join(shaderDir,`${spec.entry}-${optimize===false?'none':optimize}.wgsl`),c.wgsl);}
 }
 results.push({entry:spec.entry,sourceSha256:sha(source),variants});
 console.log(spec.entry+': '+variants.map(v=>`${v.mode} ${v.bytes} B`).join(' -> '));
}
await mkdir(dirname(destination),{recursive:true});
await writeFile(destination,JSON.stringify({compiler:COMPILER_VERSION,
 stratumRevision:'458860cb0470fa2c51516c37c1aaf34935b4e280 (CI fixture; verify your local checkout)',
 note:'Source size and CPU translation measurements only. Not GPU pipeline timing or a rendered-image equivalence test.',results},null,2)+'\n');
