export function opticalFlowPlan(){
 const buffers={},textures={},steps=[],levels=Array.from({length:5},(_,i)=>{const w=40*2**i,h=30*2**i;return {w,h,s:Math.ceil(w/32)*32};});
 for(const name of ['u','v','nu','nv','du0','dv0','du1','dv1','Ix','Iy','Iz','tmp'])buffers[name]={type:'f32',records:640*480,fill:'zero'};
 const dispatch=(entry,block,scalars,bindings,groups)=>({entry,block,scalars,bindings,groups:groups||[Math.ceil((scalars.width||scalars.w)/block[0]),Math.ceil((scalars.height||scalars.h)/block[1]),1]});
 const copy=(source,target,stride)=>({copyToTexture:{source,target,bytesPerRow:stride*4}});
 for(let i=0;i<5;i++){const {w,h,s}=levels[i];for(let j=0;j<2;j++)buffers[`image${j}_${i}`]={type:'f32',records:s*h,fill:i===4?'binary-f32':'zero',...(i===4?{source:`/reports/optical-input-${j}.bin`}:{})};for(const name of ['image0','image1','warped','flowU','flowV'])textures[`${name}Tex${i}`]={kind:'scalar-f32',dimensions:[w,h],fill:'zero',filter:'linear',addressMode:'mirror-repeat',normalizedCoords:true};}
 for(let i=4;i>=0;i--){const {w,h,s}=levels[i];for(let j=0;j<2;j++){if(i<4)steps.push(dispatch('DownscaleKernel',[32,8,1],{width:w,height:h,stride:s},{out:`image${j}_${i}`,texFine:`image${j}Tex${i+1}`}));steps.push(copy(`image${j}_${i}`,`image${j}Tex${i}`,s));}}
 let u='u',v='v',nu='nu',nv='nv';
 for(let i=0;i<5;i++){const {w,h,s}=levels[i];for(let warp=0;warp<3;warp++){
  steps.push(...['du0','dv0','du1','dv1'].map(clear=>({clear})));
  steps.push(dispatch('WarpingKernel',[32,6,1],{width:w,height:h,stride:s},{u,v,out:'tmp',texToWarp:`image1Tex${i}`}));steps.push(copy('tmp',`warpedTex${i}`,s));
  steps.push(dispatch('ComputeDerivativesKernel',[32,6,1],{width:w,height:h,stride:s},{Ix:'Ix',Iy:'Iy',Iz:'Iz',texSource:`image0Tex${i}`,texTarget:`warpedTex${i}`}));
  steps.push({repeat:250,steps:[dispatch('JacobiIteration<32,6>',[32,6,1],{w,h,s,alpha:.2},{du0:'du0',dv0:'dv0',Ix:'Ix',Iy:'Iy',Iz:'Iz',du1:'du1',dv1:'dv1'}),dispatch('JacobiIteration<32,6>',[32,6,1],{w,h,s,alpha:.2},{du0:'du1',dv0:'dv1',Ix:'Ix',Iy:'Iy',Iz:'Iz',du1:'du0',dv1:'dv0'})]});
  steps.push(dispatch('AddKernel',[256,1,1],{count:s*h},{op1:u,op2:'du0',sum:nu},[Math.ceil(s*h/256),1,1]),dispatch('AddKernel',[256,1,1],{count:s*h},{op1:v,op2:'dv0',sum:nv},[Math.ceil(s*h/256),1,1]));[u,nu]=[nu,u];[v,nv]=[nv,v];
 }
 if(i<4){const next=levels[i+1];steps.push(copy(u,`flowUTex${i}`,s),copy(v,`flowVTex${i}`,s));steps.push(dispatch('UpscaleKernel',[32,8,1],{width:next.w,height:next.h,stride:next.s,scale:next.w/w},{out:nu,texCoarse:`flowUTex${i}`}),dispatch('UpscaleKernel',[32,8,1],{width:next.w,height:next.h,stride:next.s,scale:next.h/h},{out:nv,texCoarse:`flowVTex${i}`}));[u,nu]=[nu,u];[v,nv]=[nv,v];}
 }
 return {buffers,textures,steps,preview:{kind:'image',buffer:u,width:640,height:480,format:'gray-f32'},flowOutput:{u,v}};
}
