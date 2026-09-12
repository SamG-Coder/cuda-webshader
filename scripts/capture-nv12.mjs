import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {compile} from '../src/compiler/compiler.js';

const upstream='.local/nvidia-audit/cpp/5_Domain_Specific/NV12toBGRandResize/';
const sha256=data=>createHash('sha256').update(data).digest('hex');
const width=640,height=480,batch=24,frameBytes=width*height*3*4;
const report={revision:'5443602d89ed99aede2e4b7bf329daddeadb320e',
  correctnessReference:false,
  warning:'Observed native outputs only. Original NV12 resize batch addressing leaves frames unwritten; initcheck reports uninitialized reads. Do not use these captures as a correctness oracle.',
  input:{width:1920,height:1080},output:{width,height,batch,layout:'planar BGR f32'},
  nativeSources:{},workflows:[],compilerProbes:[]};
for(const file of ['bgr_resize.cu','nv12_resize.cu','nv12_to_bgr_planar.cu','resize_convert_main.cpp','resize_convert.h']) {
  const source=(await readFile(upstream+file,'utf8')).replace(/\r\n/g,'\n');
  report.nativeSources[file]=sha256(source);
}
const input=await readFile(upstream+'data/test1920x1080.nv12');
if(input.length!==1920*1080*3/2)throw Error('Unexpected input image size');
report.input.sha256=sha256(input);
await writeFile('reports/nv12-input.bin',input);
const outputs=[];
for(const [id,name] of [['t1','resize NV12 then convert'],['t2','convert then resize BGR']]) {
  const raw=await readFile(`.local/nv12-${id}-native.bin`);
  if(raw.length!==frameBytes*batch)throw Error('Missing native frames: '+id);
  const frameHashes=Array.from({length:batch},(_,i)=>sha256(raw.subarray(i*frameBytes,(i+1)*frameBytes)));
  // Preserve every distinct frame. Upstream batch indexing and texture tile
  // boundaries mean duplicated inputs do not imply identical outputs.
  const uniqueHashes=[...new Set(frameHashes)];
  for(let j=0;j<uniqueHashes.length;j++) {
    const index=frameHashes.indexOf(uniqueHashes[j]);
    await writeFile(`reports/nv12-${id}-native-${j}.bin`,raw.subarray(index*frameBytes,(index+1)*frameBytes));
  }
  const frame=raw.subarray(0,frameBytes),values=new Float32Array(frame.buffer,frame.byteOffset,frame.length/4);
  let min=Infinity,max=-Infinity;
  for(const value of values){if(!Number.isFinite(value)||value<0||value>255)throw Error('Invalid native colour');min=Math.min(min,value);max=Math.max(max,value);}
  outputs.push(Float32Array.from(values));
  const allValues=new Float32Array(raw.buffer,raw.byteOffset,raw.length/4);
  for(const value of allValues)if(!Number.isFinite(value)||value<0||value>255)throw Error('Invalid batch colour');
  report.workflows.push({id,name,frameHashes,frameReferences:frameHashes.map(hash=>uniqueHashes.indexOf(hash)),batchSha256:sha256(raw),checkedValues:raw.length/4,firstFrameRange:{min,max},identicalFrames:uniqueHashes.length===1});
}
let differingValues=0,maxDifference=0;
for(let i=0;i<outputs[0].length;i++){const d=Math.abs(outputs[0][i]-outputs[1][i]);if(d) differingValues++;maxDifference=Math.max(maxDifference,d);}
report.processingOrderDifference={differingValues,maxDifference};
for(const [file,entry,start,end,block] of [
  ['nv12_to_bgr_planar.cu','nv12ToBGRplanarBatchKernel','__forceinline__','void nv12ToBGRplanarBatch(',[64,10,1]],
  ['nv12_resize.cu','resizeNV12BatchKernel','__global__','void resizeNV12Batch(',[32,32,1]],
  ['bgr_resize.cu','resizeBGRplanarBatchKernel','__global__','static void resizeBGRplanarBatchCore(',[32,32,1]]]) {
  const source=(await readFile(upstream+file,'utf8')).replace(/\r\n/g,'\n');
  const first=source.indexOf(start),last=source.indexOf(end,first);
  if(first<0||last<first)throw Error('Source boundary not found: '+file);
  const kernel='typedef unsigned char uint8_t;\ntypedef unsigned int uint32_t;\n'+source.slice(first,last);
  try{const result=compile(kernel,{entry,workgroupSize:block});report.compilerProbes.push({entry,compiled:true,wgslLines:result.wgsl.split('\n').length});}
  catch(error){report.compilerProbes.push({entry,compiled:false,error:error.message});}
}
await writeFile('reports/nv12-native-manifest.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({workflows:report.workflows.map(({frameHashes,...rest})=>rest),processingOrderDifference:report.processingOrderDifference,compilerProbes:report.compilerProbes},null,2));
