import {matrixFixture} from './matrixmul-fixtures.js';
import {scanUpdateFixture} from './scan-update-fixtures.js';
import {atomicCasFixture} from './atomic-cas-fixtures.js';
import {alignedCopyFixture} from './aligned-copy-fixtures.js';
import {fwtPassFixture} from './fwt-pass-fixtures.js';
import {fwtSharedFixture} from './fwt-shared-fixtures.js';
// Deterministic correctness fixtures for isolated upstream kernels, not host applications.
import {scalarFixture} from './scalar-fixtures.js';
import {blackScholesFixture} from './blackscholes-fixtures.js';
export function fixture(row) {
 if(row.entry==='fwtBatch1Kernel')return fwtSharedFixture();
 if(row.entry==='fwtBatch2Kernel')return fwtPassFixture();
 if(row.sample==='cpp/6_Performance/alignedTypes')return alignedCopyFixture(row.artifact.metadata.templateArguments.TData);
 if(row.entry==='cas_atomic')return atomicCasFixture();
 if(row.entry==='uniformUpdate')return scanUpdateFixture();
 if(row.entry.startsWith('MatrixMulCUDA<'))return matrixFixture(row.artifact.metadata.workgroupSize[0],64,64,64);
 if(row.entry==='BlackScholesGPU')return blackScholesFixture();
 if(row.entry==='scalarProdGPU')return scalarFixture({guards:0});
 const m=row.artifact.metadata,e=row.entry,two=m.workgroupSize[1]>1,transpose=e==='transposeCoalesced'||e==='transposeNoBankConflicts',w=transpose?64:32,n=two?w*w:256;
 const scalars=Object.fromEntries(m.scalars.map(s=>[s.name,({inc_value:5,b:7,n,N:n,vectorLength:n,count:n,inner_reps:3,base:19,width:w,height:w,time:0.375})[s.name]]));
 if(Object.values(scalars).some(v=>v===undefined))throw Error('Missing scalar fixture');
 const buffers=Object.fromEntries(m.bindings.map((b,k)=>{const Type=b.elementType==='i32'?Int32Array:Float32Array;return [b.name,Type.from({length:n*b.stride/4},(_,i)=>b.readOnly||['g_data','g_a','data'].includes(b.name)?(i%29-14)*(Type===Int32Array?1:0.125)+k:0)];}));
 const out=m.bindings.find(b=>!b.readOnly).name,expected=buffers[out].slice();
 for(let i=0;i<n;i++){
  const x=i%w,y=Math.floor(i/w);
  if(transpose)expected[x*w+y]=buffers.idata[i];
  else if(e==='increment_kernel')expected[i]=buffers.g_data[i]+5;
  else if(e==='kernelAddConstant')expected[i]=buffers.g_a[i]+7;
  else if(e==='incrementKernel')expected[i]=buffers.data[i]+1;
  else if(e==='incKernel')expected[i]=buffers.g_in[i]+1;
  else if(e==='SimpleKernel'||e==='copyP2PAndScale')expected[i]=buffers.src[i]*2;
  else if(e==='square_kernel')expected[i]=buffers.in[i]**2;
  else if(e==='vectorAddGPU')expected[i]=buffers.a[i]+buffers.b[i];
  else if(e==='vecAdd')expected[i]=buffers.A[i]+buffers.B[i];
  else if(e==='AddKernel')expected[i]=buffers.op1[i]+buffers.op2[i];
  else if(e==='writePatternKernel')expected[i]=19+i;
  else if(e.startsWith('updateHeightmapKernel'))expected[i]=buffers.ht[i*2+(e.endsWith('_y')?1:0)]*((x+y)%2?-1:1);
  else if(e==='calculateSlopeKernel'){if(x>0&&y>0&&x<w-1&&y<w-1){expected[i*2]=buffers.h[i+1]-buffers.h[i-1];expected[i*2+1]=buffers.h[i+w]-buffers.h[i-w];}}
  else if(e==='simple_vbo_kernel'){const u=x/w*2-1,v=y/w*2-1;expected.set([u,Math.sin(u*4+scalars.time)*Math.cos(v*4+scalars.time)*0.5,v,1],i*4);}
  else throw Error('Missing reference for '+e);
 }
 return {buffers,scalars,out,expected,groups:transpose?[2,2,1]:two?[4,4,1]:[2,1,1]};
}
