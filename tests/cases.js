/** Independent reference formulas and deterministic fixtures shared by CPU-oracle and real GPU tests. */
const SENTINEL=-9876.25;
export function randomFloats(n,seed=0x12345678){let s=seed>>>0;return Float32Array.from({length:n},()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return ((s>>>0)/4294967296)*2-1;});}
function withGuard(n,fill=0){const a=new Float32Array(n+4);a.fill(fill,0,n);a.fill(SENTINEL,n);return a;}
export function compareArrays(actual,expected,{absolute=3e-5,relative=3e-5}={}){
  if(actual.length!==expected.length)return {pass:false,message:`Length ${actual.length} != ${expected.length}`,maxAbs:Infinity};
  let maxAbs=0,maxRel=0,failures=0,first=null;
  for(let i=0;i<expected.length;i++){
    const error=Math.abs(actual[i]-expected[i]),rel=error/Math.max(Math.abs(expected[i]),1e-12);maxAbs=Math.max(maxAbs,error);maxRel=Math.max(maxRel,rel);
    if(!Number.isFinite(actual[i])||error>(expected[i]===SENTINEL?0:absolute+relative*Math.abs(expected[i]))){failures++;first ||= {index:i,actual:actual[i],expected:expected[i],error};}
  }
  return {pass:failures===0,failures,first,maxAbs,maxRel};
}
export function makeCases(){
  const cases=[];
  for(const n of [0,1,2,63,127,128,129,257,1031]){
    const x=randomFloats(Math.max(1,n)),y=withGuard(n);y.set(randomFloats(n,314159));const expected=y.slice(),a=1.75;
    for(let i=0;i<n;i++)expected[i]=a*x[i]+y[i];
    cases.push({name:`SAXPY n=${n}`,id:'saxpy',buffers:{x,y},scalars:{a,n},groups:[Math.max(1,Math.ceil(n/128))],expected:{y:expected}});
  }
  for(const n4 of [0,1,31,128,129,257]){
    const n=n4*4,x=randomFloats(Math.max(4,n)),y=withGuard(n),a=-0.625;y.set(randomFloats(n,999));const expected=y.slice();
    for(let i=0;i<n;i++)expected[i]=a*x[i]+y[i];
    cases.push({name:`float4 SAXPY records=${n4}`,id:'saxpy_vec4',buffers:{x,y},scalars:{a,n4},groups:[Math.max(1,Math.ceil(n4/128))],expected:{y:expected}});
  }
  for(const [M,N,K] of [[1,1,1],[3,5,7],[16,16,16],[17,19,23],[31,9,5],[0,5,3],[5,0,3],[5,7,0]]){
    const A=randomFloats(Math.max(1,M*K)),B=randomFloats(Math.max(1,K*N),7711),expected=withGuard(M*N);
    for(let r=0;r<M;r++)for(let c=0;c<N;c++){let sum=0;for(let k=0;k<K;k++)sum+=A[r*K+k]*B[k*N+c];expected[r*N+c]=sum;}
    for(const id of ['matmul_naive','matmul_tiled','matmul_register']){
      const tile=id==='matmul_naive'?8:16;
      cases.push({name:`${id} ${M}×${N}×${K}`,id,buffers:{A:A.slice(),B:B.slice(),C:withGuard(M*N)},scalars:{M,N,K},groups:[Math.max(1,Math.ceil(N/tile)),Math.max(1,Math.ceil(M/tile))],expected:{C:expected.slice()},tolerance:{absolute:8e-5,relative:8e-5}});
    }
  }
  for(const n of [0,1,2,127,128,129,255,256,257,1031]){
    const input=Float32Array.from({length:Math.max(1,n)},(_,i)=>((i%17)-8)*0.125),groups=Math.max(1,Math.ceil(n/256)),expected=withGuard(groups);
    for(let b=0;b<groups;b++){let sum=0;for(let i=b*256;i<Math.min((b+1)*256,n);i++)sum+=input[i];expected[b]=sum;}
    cases.push({name:`Reduction partials n=${n}`,id:'reduce_sum',buffers:{input,output:withGuard(groups)},scalars:{n},groups:[groups],expected:{output:expected},tolerance:{absolute:0,relative:0}});
  }
  for(const n of [0,1,2,3,127,128,129,259]){
    const input=randomFloats(Math.max(1,n)),expected=withGuard(n),weights=[0.0625,0.25,0.375,0.25,0.0625];
    for(let i=0;i<n;i++){let value=0;for(let k=-2;k<=2;k++)if(i+k>=0&&i+k<n)value+=input[i+k]*weights[k+2];expected[i]=value;}
    cases.push({name:`Convolution + halo n=${n}`,id:'convolution',buffers:{input,output:withGuard(n)},scalars:{n},groups:[Math.max(1,Math.ceil(n/128))],expected:{output:expected}});
  }
  for(const n of [0,1,127,128,129,1031,4099]){
    const input=Uint32Array.from({length:Math.max(1,n)},(_,i)=>(Math.imul(i+7,2654435761)^(i>>>2))>>>0),bins=new Uint32Array(260),expected=new Uint32Array(260);bins.fill(0xdecafbad,256);expected.set(bins);
    for(let i=0;i<n;i++)expected[input[i]&255]++;
    cases.push({name:`Histogram grid-stride n=${n}`,id:'histogram',buffers:{input,bins},scalars:{n},groups:[Math.max(1,Math.min(4,Math.ceil(n/128)))],expected:{bins:expected},tolerance:{absolute:0,relative:0}});
  }
  for(const [width,height] of [[1,1],[7,19],[33,35],[64,9],[0,5],[5,0]]){
    const input=randomFloats(Math.max(1,width*height)),expected=withGuard(width*height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)expected[x*height+y]=input[y*width+x];
    cases.push({name:`Transpose ${width}×${height}`,id:'transpose',buffers:{input,output:withGuard(width*height)},scalars:{width,height},groups:[Math.max(1,Math.ceil(width/32)),Math.max(1,Math.ceil(height/32))],expected:{output:expected},tolerance:{absolute:0,relative:0}});
  }
  for(const [n,dt] of [[1,0.016],[17,0.016],[129,0.033],[17,0]]){
    const position=randomFloats((n+1)*4,4141),velocity=randomFloats((n+1)*4,3939),pOut=position.slice(),vOut=velocity.slice(),time=2.5,attraction=1;
    for(let i=0;i<n;i++){
      const p=position.subarray(i*4,i*4+4),v=velocity.subarray(i*4,i*4+4),radiusSquared=p[0]*p[0]+p[2]*p[2],inv=1/Math.sqrt(radiusSquared+0.35),radial=attraction*(1.4-0.11*Math.sqrt(radiusSquared+0.001));
      const acc=[(-p[2]*0.52-p[0]*radial)*inv,Math.sin(time*0.37+p[3]*6.283185)*0.18-p[1]*0.22,(p[0]*0.52-p[2]*radial)*inv],drag=Math.max(0,1-dt*0.12);
      for(let k=0;k<3;k++){const next=(v[k]+acc[k]*dt)*drag;vOut[i*4+k]=next;pOut[i*4+k]=p[k]+next*dt;}
    }
    cases.push({name:`Particle float4 interop n=${n}, dt=${dt}`,id:'particles',buffers:{position,velocity},scalars:{n,dt,time,attraction},groups:[Math.max(1,Math.ceil(n/128))],expected:{position:pOut,velocity:vOut},tolerance:{absolute:8e-5,relative:8e-5}});
  }
  return cases;
}
export function cloneBuffers(buffers){return Object.fromEntries(Object.entries(buffers).map(([k,v])=>[k,v.slice()]));}
