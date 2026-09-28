import {readFile,writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {compileBend} from '../src/bend/compiler.js';
const [file,...args]=process.argv.slice(2);
const option=(key,fallback)=>{const i=args.indexOf(key);return i<0?fallback:args[i+1];};
try {
  if(!file)throw Error('Usage: node scripts/compile-bend.mjs program.bend --entry main --out generated/bend');
  const compiled=compileBend(await readFile(file,'utf8'),await readFile(new URL('../src/bend/runtime.cu',import.meta.url),'utf8'),{entry:option('--entry','main')});
  const out=option('--out','generated/bend');await mkdir(path.dirname(out),{recursive:true});
  await writeFile(out+'.cu',compiled.cuda);await writeFile(out+'.wgsl',compiled.artifact.wgsl);
  await writeFile(out+'.json',JSON.stringify(compiled.artifact,null,2));
  await writeFile(out+'.bend.json',JSON.stringify({parameters:compiled.program.parameters,resultType:compiled.program.resultType,revision:compiled.program.revision,verification:compiled.verification},null,2));
  console.log(`Wrote ${out}.{cu,wgsl,json,bend.json}`);
} catch(e){console.error(e.message);process.exitCode=1;}
