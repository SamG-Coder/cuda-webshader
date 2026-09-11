import {readFile,writeFile,mkdir} from 'node:fs/promises';import path from 'node:path';import {compile,serializableArtifact} from '../src/compiler/compiler.js';
const [file,...args]=process.argv.slice(2);
if(!file){console.error('Usage: node scripts/compile.mjs input.cu --entry kernelName --block 128,1,1 --out output/kernel');process.exit(2);}
const option=(name,fallback)=>{const i=args.indexOf(name);return i<0?fallback:args[i+1];};
try{
 const artifact=compile(await readFile(file,'utf8'),{entry:option('--entry',undefined),workgroupSize:option('--block','128,1,1').split(',').map(Number)});
 const base=option('--out',path.join('generated',artifact.name));await mkdir(path.dirname(base),{recursive:true});await writeFile(`${base}.wgsl`,artifact.wgsl);await writeFile(`${base}.json`,JSON.stringify(serializableArtifact(artifact),null,2));console.log(`Wrote ${base}.wgsl and ${base}.json`);
}catch(error){console.error(error.message);process.exitCode=1;}
