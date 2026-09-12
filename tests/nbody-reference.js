// Independent numerical reference for tests only; never used to render a preview.
export function nbodyInitial(n){
 const positions=new Float32Array((n+4)*4).fill(-12345),velocities=new Float32Array((n+4)*4).fill(-12345);
 for(let i=0;i<n;i++){positions.set([(i%11-5)*0.125,(i%7-3)*0.25,(i%13-6)*0.125,0.5+(i%5)*0.125],i*4);velocities.set([(i%3-1)*0.01,(i%5-2)*0.01,0,1],i*4);}
 return {positions,velocities};
}
export function nbodyStep(positions,velocities,n,dt,damping,softening){
 const next=new Float32Array(positions),velocity=new Float32Array(velocities);
 for(let i=0;i<n;i++){
  const acc=[0,0,0];for(let j=0;j<n;j++){const r=[0,1,2].map(c=>positions[j*4+c]-positions[i*4+c]),scale=positions[j*4+3]/Math.pow(softening+r.reduce((sum,x)=>sum+x*x,0),1.5);for(let c=0;c<3;c++)acc[c]+=r[c]*scale;}
  for(let c=0;c<3;c++){velocity[i*4+c]=(velocities[i*4+c]+acc[c]*dt)*damping;next[i*4+c]=positions[i*4+c]+velocity[i*4+c]*dt;}
 }
 return {positions:next,velocities:velocity};
}
