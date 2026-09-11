import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {compile, serializableArtifact} from '../src/compiler/compiler.js';
import {KERNELS} from '../src/kernels.js';
const out = new URL('../generated/', import.meta.url); await mkdir(out,{recursive:true});
for (const k of KERNELS) {
  const source=await readFile(new URL(`../kernels/${k.file}`,import.meta.url),'utf8');
  const artifact=compile(source,{entry:k.id,workgroupSize:k.workgroupSize});
  await writeFile(new URL(`${k.id}.wgsl`,out),artifact.wgsl);
  await writeFile(new URL(`${k.id}.json`,out),JSON.stringify(serializableArtifact(artifact),null,2));
  console.log(`${k.id.padEnd(20)} ${artifact.metadata.workgroupStorageBytes} shared bytes, ${artifact.wgsl.split('\n').length} WGSL lines`);
}
