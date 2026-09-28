// Independent execution oracle: upstream Bend's JS compiler, not our term machine.
// Node 24+, with --experimental-transform-types if required by upstream.
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {writeFile} from 'node:fs/promises';
import {checkBend,BEND_REVISION} from '../src/bend/compiler.js';
import {bendCases} from '../tests/bend-cases.js';
const checkout=process.argv[2];
if(!checkout)throw Error('Usage: node scripts/test-bend-reference.mjs <upstream checkout>');
assert.equal(execFileSync('git',['-C',checkout,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),BEND_REVISION,'Reference checkout revision');
execFileSync('git',['-C',checkout,'diff','--exit-code','HEAD','--','bend2']);
const Comp=await import(pathToFileURL(path.join(checkout,'bend2/comp.ts')));
const report=[];
for(const sample of bendCases) {
  const book=checkBend(sample.source);
  const code=Comp.js_lib(book,true);
  const {default:exports}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
  const {lowerBend}=await import('../src/bend/compiler.js');
  const program=lowerBend(sample.source,{entry:sample.entry});
  const result=sample.rows.map(row=>exports[sample.entry](...row.map((v,i)=>program.parameters[i].type==='Nat'?BigInt(v):v)));
  assert.deepEqual(result.map(Number),sample.expected,sample.name);
  report.push({name:sample.name,rows:result.length,passed:true});
}
await writeFile(new URL('../reports/bend-reference.json',import.meta.url),JSON.stringify({oracle:'Upstream Bend JS compiler',cases:report},null,2));
console.log(`Upstream Bend JS compiler: ${report.length} cases passed (${report.reduce((n,c)=>n+c.rows,0)} results).`);
