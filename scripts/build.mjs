import {mkdir,cp,copyFile,rm,access} from 'node:fs/promises';import {fileURLToPath} from 'node:url';
const root=new URL('../',import.meta.url),out=new URL('../dist/',import.meta.url);
try{await access(new URL('node_modules/three/build/three.webgpu.js',root));}catch{throw new Error('Run npm install first. Three.js is not installed in this checkout.');}
await rm(out,{recursive:true,force:true});await mkdir(out,{recursive:true});
for(const file of ['index.html','lab.html','src','kernels','showcases','LICENSE','THIRD_PARTY_NOTICES.md','licenses'])await cp(new URL(file,root),new URL(file,out),{recursive:true});
await mkdir(new URL('tests/',out),{recursive:true});
for(const file of ['cases.js','gpu-suite.js','three-interop.js','gpu.html','gpu-page.js'])await copyFile(new URL(`tests/${file}`,root),new URL(`tests/${file}`,out));
for(const dir of ['build/','examples/jsm/controls/'])await mkdir(new URL(`node_modules/three/${dir}`,out),{recursive:true});
for(const file of ['build/three.core.js','build/three.webgpu.js','build/three.tsl.js','examples/jsm/controls/OrbitControls.js','LICENSE'])await copyFile(new URL(`node_modules/three/${file}`,root),new URL(`node_modules/three/${file}`,out));
await mkdir(new URL('reports/',out),{recursive:true});
for(const file of ['performance-comparison.html','performance-comparison.md','comparison-summary.json','simplegl-showcase.png','live-app.png']){
  try{await access(new URL(`reports/${file}`,root));}catch{continue;}
  await copyFile(new URL(`reports/${file}`,root),new URL(`reports/${file}`,out));
}
console.log(`Static build created at ${fileURLToPath(out)}. Serve with: node scripts/serve.mjs dist`);
