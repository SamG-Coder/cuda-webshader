import {mkdir,cp,copyFile,rm,access} from 'node:fs/promises';import {fileURLToPath} from 'node:url';
const root=new URL('../',import.meta.url),out=new URL('../dist/',import.meta.url);
try{await access(new URL('node_modules/three/build/three.webgpu.js',root));}catch{throw new Error('Run npm install first. Three.js is not installed in this checkout.');}
await rm(out,{recursive:true,force:true});await mkdir(out,{recursive:true});
for(const file of ['index.html','lab.html','sandbox.html','src','kernels','showcases','LICENSE','THIRD_PARTY_NOTICES.md','licenses'])await cp(new URL(file,root),new URL(file,out),{recursive:true});
await cp(new URL('node_modules/monaco-editor/min/',root),new URL('node_modules/monaco-editor/min/',out),{recursive:true});
for(const file of ['LICENSE','ThirdPartyNotices.txt'])await copyFile(new URL(`node_modules/monaco-editor/${file}`,root),new URL(`node_modules/monaco-editor/${file}`,out));
await mkdir(new URL('tests/',out),{recursive:true});
for(const file of ['cases.js','gpu-suite.js','three-interop.js','gpu.html','gpu-page.js'])await copyFile(new URL(`tests/${file}`,root),new URL(`tests/${file}`,out));
for(const dir of ['build/','examples/jsm/controls/'])await mkdir(new URL(`node_modules/three/${dir}`,out),{recursive:true});
for(const file of ['build/three.core.js','build/three.webgpu.js','build/three.tsl.js','examples/jsm/controls/OrbitControls.js','LICENSE'])await copyFile(new URL(`node_modules/three/${file}`,root),new URL(`node_modules/three/${file}`,out));
await mkdir(new URL('reports/',out),{recursive:true});
for(const file of ['performance-comparison.html','performance-comparison.md','comparison-summary.json','simplegl-showcase.png','live-app.png','nvidia-audit.json','nvidia-audit.md','nvidia-gpu.json','nvidia-native.json','nvidia-transpose.json','nvidia-transpose-native.txt','nvidia-scalar.json','nvidia-scalar-native.txt','nvidia-blackscholes.json','nvidia-blackscholes-native.txt','nvidia-matrixmul.json','nvidia-matrixmul-native.txt','nvidia-scan-update.json','nvidia-scan-update-native.txt','nvidia-atomic-cas.json','nvidia-atomic-cas-native.txt','nvidia-aligned-copy.json','nvidia-aligned-copy-native.txt','nvidia-fwt-pass.json','nvidia-fwt-pass-native.txt','nvidia-fwt-shared.json','nvidia-fwt-shared-native.txt','nvidia-histogram-merge.json','nvidia-histogram-merge-native.txt','shared-wrapper-native.txt','vector-initializers-native.txt','helper-pointers-native.txt','shared-helpers-native.txt','constant-globals-native.txt','type-traits-native.txt','helper-templates-native.txt','nvidia-bitonic.json','nvidia-bitonic-native.txt','nvidia-driver-add.json','nvidia-driver-add-native.txt','nvidia-mpi-sqrt.json','nvidia-mpi-sqrt-native.txt','nvidia-inverse-cnd.json','nvidia-inverse-cnd-native.txt']){
  try{await access(new URL(`reports/${file}`,root));}catch{continue;}
  await copyFile(new URL(`reports/${file}`,root),new URL(`reports/${file}`,out));
}
console.log(`Static build created at ${fileURLToPath(out)}. Serve with: node scripts/serve.mjs dist`);
