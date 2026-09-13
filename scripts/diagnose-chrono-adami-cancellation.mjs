// Independent double-precision diagnostic, never used by the GPU simulation.
import {readFileSync,writeFileSync} from 'node:fs';
const n=30327,total=794643,marker=23513,b=readFileSync('reports/chrono-adami-input.bin');let offset=0;
const take=(count,width,integer=false)=>{const out=Array.from({length:count},(_,i)=>Array.from({length:width},(_,j)=>b[integer?'readUInt32LE':'readFloatLE'](offset+(i*width+j)*4)));offset+=count*width*4;return out;};
const offsets=take(n+1,1,true).flat(),ids=take(total,1,true).flat(),pos=take(n,4),rho=take(n,4);
const p=JSON.parse(readFileSync('reports/chrono-params.json')),get=name=>p['constant.paramsD.'+name];let sumW=0,sumP=0,sumG=0,scale=0,fluidNeighbors=0;
for(let k=offsets[marker]+1;k<offsets[marker+1];k++){
 const j=ids[k];if(rho[j][3]>-.5)continue;fluidNeighbors++;
 let d=pos[marker].slice(0,3).map((v,a)=>v-pos[j][a]);
 for(const [a,axis]of [...'xyz'].entries())if(get(axis+'_periodic')){const period=get('boxDims.'+axis);d[a]-=period*Math.round(d[a]/period);}
 if(d.reduce((sum,v)=>sum+v*v,0)<(get('epsMinMarkersDis')*get('h'))**2)d=[get('epsMinMarkersDis')*get('h'),0,0];
 const q=Math.hypot(...d)*get('ooh'),alpha=.31830988618379*get('ooh')**3/4,w=q<1?alpha*((2-q)**3-4*(1-q)**3):q<2?alpha*(2-q)**3:0;
 if(get('kernel_type')!==1)throw Error('Diagnostic requires cubic spline');
 const pw=rho[j][1]*w,g=rho[j][0]*d.reduce((sum,v,a)=>sum+get('gravity.'+'xyz'[a])*v,0)*w;
 sumW+=w;sumP+=pw;sumG+=g;scale+=Math.abs(pw)+Math.abs(g);
}
const result={marker,fluidNeighbors,pressureContribution:sumP/sumW,gravityContribution:sumG/sumW,doubleReference:(sumP+sumG)/sumW,absoluteTermScale:scale/sumW};
writeFileSync('reports/chrono-adami-cancellation.json',JSON.stringify(result,null,2)+'\n');console.log(result);
