export const fdtdCoefficients=[0.5,0.0625,0.03125,0.015625,0.0078125];
export function fdtdInitial(x,y,z){return Float32Array.from({length:(x+8)*(y+8)*(z+8)},(_,i)=>(i%17-8)*0.125);}
export function fdtdStep(input,x,y,z,coeff=fdtdCoefficients){
 const out=new Float32Array(input),sy=x+8,sz=sy*(y+8);
 for(let iz=4;iz<z+4;iz++)for(let iy=4;iy<y+4;iy++)for(let ix=4;ix<x+4;ix++){
  const p=iz*sz+iy*sy+ix;let value=input[p]*coeff[0];for(let r=1;r<=4;r++)value+=coeff[r]*(input[p-r]+input[p+r]+input[p-r*sy]+input[p+r*sy]+input[p-r*sz]+input[p+r*sz]);out[p]=value;
 }return out;
}
