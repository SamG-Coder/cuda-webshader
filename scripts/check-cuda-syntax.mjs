// Optional extra check: an ordinary C++ compiler parses all .cu samples with CUDA declarations stubbed.
// This catches invalid C++ syntax/types, NOT CUDA launch semantics, shader validity or GPU behavior.
import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';import {KERNELS} from '../src/kernels.js';
const root=fileURLToPath(new URL('../',import.meta.url)),compiler=process.env.CXX||'g++';let failed=0;
for(const k of KERNELS){const r=spawnSync(compiler,['-std=c++17','-x','c++','-fsyntax-only','-include','tests/cuda-syntax-stubs.hpp',`kernels/${k.file}`],{cwd:root,encoding:'utf8'});if(r.error){console.error(`${compiler} unavailable: ${r.error.message}`);process.exit(1);}if(r.status!==0){failed++;console.error(`FAIL ${k.file}\n${r.stderr}`);}else console.log(`PASS C++ syntax only: ${k.file}`);}
console.log(`${KERNELS.length-failed}/${KERNELS.length} C++ syntax checks passed. No GPU execution performed.`);process.exitCode=failed?1:0;
