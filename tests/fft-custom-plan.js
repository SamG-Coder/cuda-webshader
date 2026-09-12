export function fftCustomPlan(variant,test=1){
 const width=test?2000:37,height=test?2000:19,fw=test?2048:64,fh=test?2048:32,n=fw*fh,compact=n/2,wide=fh*(fw/2+16),buffers={},textures={};
 const add=(name,type,records)=>buffers[name]={type,records,fill:'zero'};
 add('input','f32',width*height);add('kernel','f32',42);add('paddedData','f32',n);add('paddedKernel','f32',n);add('data0','vec2<f32>',compact);add('kernel0','vec2<f32>',compact);
 for(const name of ['input','kernel']){buffers[name].fill='binary-f32';buffers[name].source=`reports/fft-convolution-${test}-${name}.bin`;textures[name+'Texture']={kind:'linear-f32',records:buffers[name].records,source:buffers[name].source};}
 const tex=(name,records)=>textures[name]={kind:'linear-f32',components:2,records,fill:'zero'};
 tex('compactTexture',compact);if(variant===1){add('dataWide','vec2<f32>',wide);add('kernelWide','vec2<f32>',wide);tex('wideTexture',wide);}else {tex('kernelTextureComplex',compact);add('result0','vec2<f32>',compact);}
 const scalars={fftH:fh,fftW:fw,kernelH:7,kernelW:6,kernelY:3,kernelX:4},steps=[{entry:'padKernel_kernel',block:[32,8,1],groups:[1,1,1],bindings:{d_Dst:'paddedKernel',d_Src:'kernel',texFloat:'kernelTexture'},scalars},{entry:'padDataClampToBorder_kernel',block:[32,8,1],groups:[Math.ceil(fw/32),Math.ceil(fh/8),1],bindings:{d_Dst:'paddedData',d_Src:'input',texFloat:'inputTexture'},scalars:{...scalars,dataH:height,dataW:width}}];
 for(const [source,target]of [['paddedKernel','kernel0'],['paddedData','data0']])steps.push({copyBytes:{source,target,byteLength:n*4}},{complexFFT:{source:target,target,width:fw/2,height:fh}});
 const threadCount=fh*fw/4,phaseBase=-Math.PI/(fw/2),dispatch=(entry,bindings,scalars)=>steps.push({entry,block:[256,1,1],groups:[Math.ceil(threadCount/256),1,1],bindings,scalars});
 if(variant===1){
  for(const [source,target]of [['kernel0','kernelWide'],['data0','dataWide']]){steps.push({copyToLinearTexture:{source,target:'compactTexture'}});dispatch('spPostprocess2D_kernel',{d_Dst:target,d_Src:source,texComplex:'compactTexture'},{DY:fh,DX:fw/2,threadCount,padding:16,phaseBase});}
  steps.push({entry:'modulateAndNormalize_kernel',block:[256,1,1],groups:[Math.ceil(wide/256),1,1],bindings:{d_Dst:'dataWide',d_Src:'kernelWide'},scalars:{dataSize:wide,c:1/n}},{copyToLinearTexture:{source:'dataWide',target:'wideTexture'}});
  dispatch('spPreprocess2D_kernel',{d_Dst:'data0',d_Src:'dataWide',texComplex:'wideTexture'},{DY:fh,DX:fw/2,threadCount,padding:16,phaseBase});
 }else{
  steps.push({copyToLinearTexture:{source:'data0',target:'compactTexture'}},{copyToLinearTexture:{source:'kernel0',target:'kernelTextureComplex'}});
  // Separate output avoids WebGPU writable-binding aliases; original texture reads are unchanged.
  dispatch('spProcess2D_kernel',{d_Dst:'result0',d_SrcA:'data0',d_SrcB:'kernel0',texComplexA:'compactTexture',texComplexB:'kernelTextureComplex'},{DY:fh,DX:fw/2,threadCount,phaseBase,c:1/n});
 }
 steps.push({complexFFT:{source:variant===1?'data0':'result0',target:'data0',width:fw/2,height:fh,inverse:true}},{copyBytes:{source:'data0',target:'paddedData',byteLength:n*4}});
 return {buffers,textures,steps,preview:{kind:'image',buffer:'paddedData',format:'gray-f32',width:fw,height:fh,range:[0,5055]}};
}
