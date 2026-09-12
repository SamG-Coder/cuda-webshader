import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';
const source=readFileSync(new URL('quadtree-storage-references.cu',import.meta.url),'utf8');
test('Class pointers into value buffers preserve chained typed offsets',()=>{const a=compile(source,{entry:'write_alias_nodes',valueBuffers:['nodes'],workgroupSize:[32]});assert.match(a.wgsl,/var cw_offset_/);assert.doesNotMatch(a.wgsl,/cw_heap_Quadtree_node/);});
test('Value-buffer aliases retain const protection',()=>{assert.throws(()=>compile(source.replace('Quadtree_node *base=&nodes[offset];','const Quadtree_node *base=&nodes[offset];'),{entry:'write_alias_nodes',valueBuffers:['nodes'],workgroupSize:[32]}),/discard const/);});
