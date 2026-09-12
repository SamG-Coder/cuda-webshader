import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validatePipeline,executePipeline} from '../src/sandbox/pipeline.js';
const preset=()=>JSON.parse(readFileSync('showcases/marching-cubes/pipeline.json','utf8')).pipeline;
test('pipeline rejects unsafe allocation, scan aliasing, control offsets and invalid mesh declarations',()=>{
 for(const change of [p=>p.buffers.pos.records=1048577,p=>p.steps[1].scan.target='voxelVerts',p=>p.steps[1].scan.count=4097,p=>p.steps[4].groups[0].index=1,p=>p.steps[0].block=[1024,2,1],p=>p.preview.positions='volume',p=>p.preview.count={buffer:'vertexTotal',divideCeil:0}]){const plan=preset();change(plan);assert.throws(()=>validatePipeline(plan));}
 assert.ok(validatePipeline(preset())>0);
});
test('pipeline invalidates GPU control totals after a write and rejects overflowing draw counts',async()=>{
 const plan={buffers:{total:{type:'u32',records:1,fill:'zero'},pos:{type:'vec4<f32>',records:3,fill:'zero'},norm:{type:'vec4<f32>',records:3,fill:'zero'}},steps:[{entry:'write',block:[1,1,1],groups:[1,1,1],scalars:{n:3}},{entry:'write',block:[1,1,1],groups:[{buffer:'total'},1,1],scalars:{n:6}}],preview:{kind:'triangles',positions:'pos',normals:'norm',count:{buffer:'total'}}};
 let reads=0;const dispatched=[],artifact={name:'write',metadata:{workgroupSize:[1,1,1],bindings:[{name:'total',elementType:'u32',readOnly:false}]}},runtime={createBuffer:data=>({data}),kernel:async()=>({bind:(buffers,scalars)=>({buffers,scalars})}),batch:()=>({dispatch(invocation,groups){dispatched.push(groups);invocation.buffers.total.data[0]=invocation.scalars.n;return this;},submit(){}}),idle:async()=>{},read:async buffer=>{reads++;return buffer.data;}};
 await assert.rejects(executePipeline({plan,source:'',defines:{},compiler:{},runtime,rootArtifact:artifact,resources:[],textureResources:[]}),/Triangle count/);
 assert.deepEqual(dispatched,[[1,1,1],[3,1,1]]);assert.equal(reads,2);
});

test('Image pipeline rejects oversized dimensions, wrong formats and insufficient storage',()=>{const plan=JSON.parse(readFileSync('showcases/box-filter/pipeline.json','utf8')).pipeline;assert.ok(validatePipeline(plan)>0);for(const change of [p=>p.preview.width=4097,p=>p.preview.height=0,p=>p.preview.format='gray-f32',p=>p.buffers.output.records=10,p=>p.buffers.output.type='f32']){const p=structuredClone(plan);change(p);assert.throws(()=>validatePipeline(p),/Image preview/);}});
