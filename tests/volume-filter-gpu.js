export async function checkVolumeFilter(runtime){
 const source=await(await fetch('/tests/volume-filter-kernel.cuh')).text(),kernel=await runtime.kernel(source,{entry:'d_filter_surface3d',defines:{__CUDACC__:1},workgroupSize:[8,8,1]}),comparisons=[];
 for(const bucky of [false,true]){
  const width=bucky?32:8,height=width,depth=bucky?32:4,total=width*height*depth;
  const data=bucky?new Uint8Array(await(await fetch('/showcases/volume-filter/Bucky.raw')).arrayBuffer()):Uint8Array.from({length:total},(_,i)=>(i*37+11)&255);
  for(let scenario=0;scenario<4;scenario++){
   const normalizedCoords=scenario<2,filter=scenario===3?'nearest':'linear',addressMode=normalizedCoords?'repeat':'clamp-to-edge',options={width,height,depth,normalizedCoords,filter,addressMode};
   const input=runtime.createTexture3D(data,options),output=[0,1].map(()=>runtime.createTexture3D(null,{...options,format:'rgba8unorm',storage:true}));
   try{
    const scalars={filterSize:scenario?3:1,filter_offset:scenario?.0625:0,'volumeSize.width':width,'volumeSize.height':height,'volumeSize.depth':depth},weights=scenario?[[-.125,0,0,.25],[0,.25,0,.5],[.125,0,.25,.25]]:[[0,0,0,1]];
    for(let i=0;i<weights.length;i++)for(let c=0;c<4;c++)scalars['constant.c_filterData['+i+'].'+'xyzw'[c]]=weights[i][c];
    const before={...runtime.stats},groups=[width/8,height/8,depth];runtime.batch().dispatch(kernel.bind({volumeTexIn:input,volumeTexOut:output[0]},scalars),groups).dispatch(kernel.bind({volumeTexIn:output[0],volumeTexOut:output[1]},scalars),groups).submit();await runtime.idle();
    if(runtime.stats.dataBytesUploaded!==before.dataBytesUploaded||runtime.stats.readbackBytes!==before.readbackBytes)throw Error('Intermediate volume transferred through CPU');
    for(let pass=0;pass<2;pass++){
     const native=new Uint8Array(await(await fetch('/reports/volume-filter-'+(bucky?'bucky-':'')+'native-'+scenario+'-'+pass+'.bin')).arrayBuffer());if(native.length!==total)throw Error('Native volume size');
     let mismatches=0,maxError=0;for(let slice=0;slice<depth;slice++){const result=await runtime.readTextureSlice(output[pass],{slice});if(result.width!==width||result.height!==height)throw Error('Slice dimensions');for(let i=0;i<result.data.length;i++){const actual=Math.round(result.data[i]*255),expected=native[slice*width*height+i];if(actual!==expected)mismatches++;maxError=Math.max(maxError,Math.abs(actual-expected));}}
     comparisons.push({dataset:bucky?'Bucky':'synthetic',scenario,pass,voxels:total,mismatches,maxError,noIntermediateCPUTransfer:true});
    }
    for(const slice of [-1,depth,.5]){let rejected=false;try{await runtime.readTextureSlice(output[0],{slice});}catch{rejected=true;}if(!rejected)throw Error('Invalid slice accepted');}
   }finally{runtime.destroyTexture(input);output.forEach(t=>runtime.destroyTexture(t));}
  }
 }
 if(comparisons.some(c=>c.mismatches))throw Error(JSON.stringify(comparisons));return {comparisons};
}
