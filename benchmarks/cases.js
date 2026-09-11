// Shared, deterministic inputs and independent host references for both APIs.
import {randomFloats} from '../tests/cases.js';
export function makeComparisonCases(scale='small') {
  const n=scale==='small'?262144:4194304, m=scale==='small'?128:512;
  const result=[];
  const add=(id,buffers,scalars,expected,groups,block=[128,1,1])=>result.push({key:`${scale}-${id}`,id,buffers,scalars,expected,groups,block,iterations:id.startsWith('matmul')?512:2048,samples:9,warmup:5});
  const x=randomFloats(n,31),y=randomFloats(n,17),out=Float32Array.from(y,(v,i)=>v+0.125*x[i]);
  add('saxpy',{x,y},{a:0.125,n},{y:out},[Math.ceil(n/128)]);
  add('saxpy_vec4',{x,y},{a:0.125,n4:n/4},{y:out},[Math.ceil(n/512)]);
  const A=randomFloats(m*m,234),B=randomFloats(m*m,879),C=new Float32Array(m*m);
  for(let r=0;r<m;r++)for(let c=0;c<m;c++){let sum=0;for(let k=0;k<m;k++)sum+=A[r*m+k]*B[k*m+c];C[r*m+c]=sum;}
  for(const id of ['matmul_naive','matmul_tiled','matmul_register']){const tile=id==='matmul_naive'?8:16;add(id,{A,B,C:new Float32Array(m*m)},{M:m,N:m,K:m},{C},[m/tile,m/tile],id==='matmul_tiled'?[16,16,1]:[8,8,1]);}
  const input=Float32Array.from({length:n},(_,i)=>(i%17-8)*0.125);
  add('reduce_sum',{input},{n},{output:new Float32Array([input.reduce((a,b)=>a+b,0)])},[Math.ceil(n/256)]);
  const conv=new Float32Array(n),w=[0.0625,0.25,0.375,0.25,0.0625];
  for(let i=0;i<n;i++)for(let j=-2;j<=2;j++)if(i+j>=0&&i+j<n)conv[i]+=x[i+j]*w[j+2];
  add('convolution',{input:x,output:new Float32Array(n)},{n},{output:conv},[Math.ceil(n/128)]);
  const ints=Uint32Array.from({length:n},(_,i)=>(Math.imul(i+7,2654435761)^(i>>>2))>>>0),bins=new Uint32Array(256);
  for(const v of ints)bins[v&255]++;
  add('histogram',{input:ints,bins:new Uint32Array(256)},{n},{bins},[256]);
  const width=scale==='small'?512:2048,height=n/width,t=new Float32Array(n);
  for(let r=0;r<height;r++)for(let c=0;c<width;c++)t[c*height+r]=x[r*width+c];
  add('transpose',{input:x,output:new Float32Array(n)},{width,height},{output:t},[width/32,height/32],[32,8,1]);
  const pn=scale==='small'?65536:262144,position=randomFloats(pn*4,4141),velocity=randomFloats(pn*4,3939),po=position.slice(),vo=velocity.slice(),dt=Math.fround(0.016),time=2.5,attraction=1;
  for(let i=0;i<pn;i++){const j=i*4,p=position.subarray(j,j+4),v=velocity.subarray(j,j+4),r=p[0]*p[0]+p[2]*p[2],inv=1/Math.sqrt(r+0.35),rad=1.4-0.11*Math.sqrt(r+0.001),acc=[(-p[2]*0.52-p[0]*rad)*inv,Math.sin(time*0.37+p[3]*6.283185)*0.18-p[1]*0.22,(p[0]*0.52-p[2]*rad)*inv];for(let k=0;k<3;k++){vo[j+k]=(v[k]+acc[k]*dt)*(1-dt*0.12);po[j+k]=p[k]+(v[k]+acc[k]*dt)*(1-dt*0.12)*dt;}}
  add('particles',{position,velocity},{n:pn,dt,time,attraction},{position:po,velocity:vo},[pn/128]);
  return result;
}
