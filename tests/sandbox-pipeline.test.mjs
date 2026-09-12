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

test('Float image pipelines validate copy storage and preview dimensions',()=>{const preset=()=>JSON.parse(readFileSync('showcases/dct/pipeline.json','utf8')).pipeline;assert.ok(validatePipeline(preset())>0);for(const change of [p=>p.buffers.working.type='u32',p=>p.steps[1].copyToTexture.target='missing',p=>p.buffers.working.records=10,p=>p.textures.plane.dimensions=[512,0],p=>p.textures.plane.fill='random',p=>p.preview.width=513,p=>p.buffers.output.type='u32']){const p=preset();change(p);assert.throws(()=>validatePipeline(p));}});

test('Ocean FFT pipeline rejects incomplete complex buffers and invalid transform shapes',()=>{const preset=()=>JSON.parse(readFileSync('showcases/ocean/pipeline.json','utf8')).pipeline;assert.ok(validatePipeline(preset())>0);for(const change of [p=>p.steps[1].inverseFFT.width=255,p=>p.buffers.ht.type='f32',p=>p.buffers.ht.records=32,p=>p.steps[0].scalars.t={time:false}]){const plan=preset();change(plan);assert.throws(()=>validatePipeline(plan));}});

test('Particle pipelines reject aliasing sorts and invalid particle previews',()=>{const preset=()=>JSON.parse(readFileSync('showcases/particle-collision/pipeline.json','utf8')).pipeline;assert.ok(validatePipeline(preset())>0);for(const change of [p=>p.steps[2].sortPairs.values='hash',p=>p.steps[2].sortPairs.count=1025,p=>p.preview.radius=0,p=>p.preview.count=1025,p=>p.preview.positions='hash']){const plan=preset();change(plan);assert.throws(()=>validatePipeline(plan));}});

test('Float depth sorts require explicit f32 keys and uint payload buffers',()=>{
 const preset=()=>JSON.parse(readFileSync('showcases/particle-collision/pipeline.json','utf8')).pipeline;
 const valid=preset();valid.buffers.hash.type='f32';valid.steps[2].sortPairs.keyType='f32';assert.ok(validatePipeline(valid)>0);
 for(const change of [p=>delete p.steps[2].sortPairs.keyType,p=>p.steps[2].sortPairs.keyType='float',p=>p.steps[2].sortPairs.keyType=null,p=>p.buffers.index.type='f32',p=>p.steps[2].sortPairs.keys='missing']){
  const plan=structuredClone(valid);change(plan);assert.throws(()=>validatePipeline(plan));
 }
});

test('Pipeline feedback copies reject aliases, type mismatch and invalid record ranges',()=>{
 const plan={buffers:{a:{type:'vec4<f32>',records:4,fill:'zero'},b:{type:'vec4<f32>',records:4,fill:'zero'}},steps:[{copyBuffer:{source:'a',target:'b',records:4}}],preview:{kind:'particles',positions:'a',count:4,radius:.1}};
 assert.ok(validatePipeline(plan)>0);
 for(const change of [p=>p.steps[0].copyBuffer.target='a',p=>p.steps[0].copyBuffer.source='missing',p=>p.buffers.b.type='f32',p=>p.steps[0].copyBuffer.records=5,p=>p.steps[0].copyBuffer.records=0,p=>p.steps[0].copyBuffer.records=1.5]){
  const p=structuredClone(plan);change(p);assert.throws(()=>validatePipeline(p),/Buffer copy/);
 }
});

test('Float4 pipeline volumes reject invalid dimensions and excessive allocation before fetching',async()=>{
 const buffers={a:{type:'vec4<f32>',records:4,fill:'zero'},b:{type:'vec4<f32>',records:4,fill:'zero'}};
 for(const texture of [{dimensions:[64,64]},{dimensions:[64,0,64]},{dimensions:[2048,2048,2048]},{dimensions:[64,64,64],storage:true}]){
  const plan={buffers,steps:[{copyBuffer:{source:'a',target:'b',records:4}}],textures:{noise:{kind:'volume-float4',source:'never-fetch',...texture}},preview:{kind:'particles',positions:'a',count:4,radius:.1}};
  await assert.rejects(executePipeline({plan,source:'',compiler:{},runtime:{createBuffer:data=>({data})},rootArtifact:{},resources:[],textureResources:[]}),/Float4 volume|64 MiB/);
 }
});

test('Smoke previews require bounded typed buffers and explicit sorting steps',()=>{
 const preset=()=>JSON.parse(readFileSync('showcases/smoke/pipeline.json','utf8')).pipeline;
 assert.ok(validatePipeline(preset())>0);
 for(const change of [p=>p.preview.count=20000,p=>p.preview.radius=0,p=>p.preview.velocities='indices',p=>p.preview.depthStep=50,p=>p.preview.indexStep=3,p=>p.preview.indices='missing']){
  const plan=preset();change(plan);assert.throws(()=>validatePipeline(plan),/Smoke preview/);
 }
});

test('Real FFT pipeline validates padded real rows and packed complex spectra',()=>{
 const preset=()=>({buffers:{real:{type:'f32',records:40,fill:'zero'},frequency:{type:'vec2<f32>',records:20,fill:'zero'}},steps:[{realFFT:{source:'real',target:'frequency',width:8,height:4,realStride:10}}],preview:{kind:'image',format:'gray-f32',buffer:'real',width:8,height:4}});
 assert.ok(validatePipeline(preset())>0);
 for(const change of [p=>p.steps[0].realFFT.width=7,p=>p.steps[0].realFFT.inverse=true,p=>p.steps[0].realFFT.realStride=7,p=>p.steps[0].realFFT.realStride=null,p=>p.buffers.real.records=39,p=>p.buffers.frequency.records=19,p=>p.buffers.frequency.type='f32']){
  const p=preset();change(p);assert.throws(()=>validatePipeline(p),/Real FFT/);
 }
 const p=preset();Object.assign(p.steps[0].realFFT,{source:'frequency',target:'real',inverse:true});assert.ok(validatePipeline(p)>0);
});

test('Fluid preview rejects invalid particle counts, force controls and texture pitches',()=>{const fluid=()=>JSON.parse(readFileSync('showcases/fluids/pipeline.json','utf8')).pipeline;assert.ok(validatePipeline(fluid())>0);for(const change of [p=>p.preview.count=262145,p=>p.preview.positions='realX',p=>p.preview.stir.step=99,p=>p.preview.stir.radius=256,p=>p.preview.stir.scale=Infinity,p=>p.preview.stir.fx='absent',p=>p.steps[1].copyToTexture.bytesPerRow=4095,p=>p.steps[1].copyToTexture.bytesPerRow=4088,p=>p.textures.field.kind='scalar-f32']){const p=fluid();change(p);assert.throws(()=>validatePipeline(p));}});

test('Signed disparity display validates range and output storage',()=>{const preset=()=>JSON.parse(readFileSync('showcases/stereo/pipeline.json','utf8')).pipeline;assert.ok(validatePipeline(preset())>0);for(const range of [null,[0,0],[0,Infinity],[0]]){const p=preset();p.preview.range=range;assert.throws(()=>validatePipeline(p));}const p=preset();p.buffers.g_odata.type='f32';assert.throws(()=>validatePipeline(p));});
