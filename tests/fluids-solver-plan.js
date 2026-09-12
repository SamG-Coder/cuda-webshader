export function fluidsSolverPlan(width){
 const height=width,pitch=width+(width===64?3:0),realStride=2*(Math.floor(width/2)+1),packed=Math.floor(width/2)+1;
 const kernel=(entry,bindings,scalars,groups=[Math.ceil(width/64),Math.ceil(height/64),1],block=[64,4,1])=>({entry,bindings,scalars,groups,block});
 return {buffers:{
 velocity:{type:'vec2<f32>',records:pitch*height,fill:'binary-f32',source:'/reports/fluids-solver-'+width+'-input-velocity.bin'},
 particles:{type:'vec2<f32>',records:width*height,fill:'binary-f32',source:'/reports/fluids-solver-'+width+'-input-particles.bin'},
 realX:{type:'f32',records:realStride*height,fill:'zero'},realY:{type:'f32',records:realStride*height,fill:'zero'},
 complexX:{type:'vec2<f32>',records:packed*height,fill:'zero'},complexY:{type:'vec2<f32>',records:packed*height,fill:'zero'}
 },textures:{field:{kind:'vector2-f32',dimensions:[width,height],fill:'zero',filter:'linear',addressMode:'clamp-to-edge',normalizedCoords:false}},
 steps:[
 kernel('addForces_k',{v:'velocity'},{dx:width,dy:height,spx:width/2-4,spy:height/2-4,fx:.06,fy:.04,r:4,pitch:pitch*8},[1,1,1],[9,9,1]),
 {copyToTexture:{source:'velocity',target:'field',bytesPerRow:pitch*8}},
 kernel('advectVelocity_k',{v:'velocity',vx:'realX',vy:'realY',texObject:'field'},{dx:width,pdx:realStride,dy:height,dt:.09,lb:16}),
 ...['X','Y'].map(c=>({realFFT:{source:'real'+c,target:'complex'+c,width,height,realStride}})),
 kernel('diffuseProject_k',{vx:'complexX',vy:'complexY'},{dx:packed,dy:height,dt:.09,visc:.0025,lb:16},[Math.ceil(packed/64),Math.ceil(height/64),1]),
 ...['X','Y'].map(c=>({realFFT:{source:'complex'+c,target:'real'+c,width,height,realStride,inverse:true}})),
 kernel('updateVelocity_k',{v:'velocity',vx:'realX',vy:'realY'},{dx:width,pdx:realStride,dy:height,lb:16,pitch:pitch*8}),
 kernel('advectParticles_k',{part:'particles',v:'velocity'},{dx:width,dy:height,dt:.09,lb:16,pitch:pitch*8})
 ],preview:{kind:'image',buffer:'realX',format:'gray-f32',width:realStride,height}};
}
