import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {mouseCameraPlan} from '../src/sandbox/mouse-camera.js';
test('mouse camera preserves original render quality and pipeline',()=>{
 const original=JSON.parse(readFileSync('showcases/pathtracer/pipeline.json')).pipeline,before=JSON.stringify(original),plan=mouseCameraPlan(original);
 assert.equal(JSON.stringify(original),before);assert.equal(plan.preview.width,1200);assert.equal(plan.preview.height,800);assert.equal(plan.steps.find(s=>s.entry==='render').scalars.ns,10);
 assert.deepEqual(plan.steps.filter(s=>s.entry!=='sandbox_mouse_camera'),original.steps);
 for(const [name,buffer]of Object.entries(original.buffers))assert.deepEqual(plan.buffers[name],buffer);
 assert.deepEqual(mouseCameraPlan(plan),plan);
});
