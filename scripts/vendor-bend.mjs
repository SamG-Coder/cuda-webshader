// Regenerate the browser frontend from the pinned upstream checkout (Node 24+).
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {stripTypeScriptTypes} from 'node:module';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
const dir = process.argv[2];
if (!dir) throw Error('Usage: node scripts/vendor-bend.mjs <bend checkout>');
const revision = '3378e6237ed431d17629efd36d24c96241815b7e';
const out = new URL('../src/bend/vendor/', import.meta.url);
await mkdir(out, {recursive:true});
const normalized=text=>text.replace(/\r\n/g,'\n');
async function pinned(file) {
  const text=normalized(await readFile(path.join(dir,file),'utf8'));
  const original=normalized(execFileSync('git',['-C',dir,'show',revision+':'+file],{encoding:'utf8',maxBuffer:2*1024*1024}));
  if(text!==original)throw Error(`${file} does not match pinned Bend revision ${revision}`);
  return text;
}
const source = await pinned('bend2/bend.ts');
// Remove only the filesystem/package loader. Parser, theory and checker stay intact.
let portable = source.replace(/^import \* as \w+ from "node:[^"]+";\r?\n/gm, '');
const start = portable.indexOf('export const BEND_DIR =');
const end = portable.indexOf('// Tele', start);
if (start < 0 || end < 0) throw Error('Upstream loader boundary changed');
portable = portable.slice(0, start) + portable.slice(end);
portable = stripTypeScriptTypes(portable, {mode:'strip'});
portable = portable.replace(/^[ \t]+$/gm,'').replace(/^(\/\/.*?)[ \t]+$/gm,'$1');
const attribution='// Copyright 2026 HigherOrderCO. Apache-2.0.\n// Source: https://github.com/bendlang/bend/tree/'+revision+'\n// License and source hashes: LICENSE and provenance.json in this directory.\n';
await writeFile(new URL('bend.js', out), attribution+'// Modified for CUDA WebShader: TypeScript types and filesystem/package loader removed.\n'+portable);
const base = await pinned('bend2/base.bend');
await writeFile(new URL('base.js', out), attribution+'// Modified for CUDA WebShader: upstream base.bend wrapped as an ES module string (LF line endings).\nexport default '+JSON.stringify(base)+';\n');
await writeFile(new URL('LICENSE', out), await pinned('LICENSE'));
await writeFile(new URL('provenance.json', out), JSON.stringify({repository:'https://github.com/bendlang/bend', revision, files:{'bend2/bend.ts':createHash('sha256').update(source).digest('hex'),'bend2/base.bend':createHash('sha256').update(base).digest('hex')}, modification:'TypeScript types and filesystem/package loader removed; parser and checker unchanged.'}, null, 2)+'\n');
