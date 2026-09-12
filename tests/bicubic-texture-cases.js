export const bicubicEntries=['d_render','d_render','d_renderBicubic','d_renderFastBicubic','d_renderCatRom'];
export function bicubicCase(scenario){const [width,height,textureWidth,textureHeight,scale,tx,ty,cx,cy]=[[35,19,16,12,.75,-.3125,.625,8,6],[128,96,512,512,3.25,-10.375,-20.625,64,48],[512,512,512,512,.35,0,0,256,256]][scenario];return {width,height,textureWidth,textureHeight,scalars:{width,height,scale,tx,ty,cx,cy}};}
export function bicubicInput(){return Uint8Array.from({length:16*12},(_,i)=>64+(i*37)%128);}
export function bicubicReference(bytes,textureWidth,textureHeight,width,height,scalars,mode){
 const at=(x,y)=>bytes[Math.max(0,Math.min(textureHeight-1,y))*textureWidth+Math.max(0,Math.min(textureWidth-1,x))]/255;
 const linear=(x,y)=>{x-=.5;y-=.5;const px=Math.floor(x),py=Math.floor(y),fx=x-px,fy=y-py;return (at(px,py)*(1-fx)+at(px+1,py)*fx)*(1-fy)+(at(px,py+1)*(1-fx)+at(px+1,py+1)*fx)*fy;};
 const weights=(a,cat)=>cat?[-.5*a+a*a-.5*a*a*a,1-2.5*a*a+1.5*a*a*a,.5*a+2*a*a-1.5*a*a*a,-.5*a*a+.5*a*a*a]:[(1-a)**3/6,(3*a*a*a-6*a*a+4)/6,(-3*a*a*a+3*a*a+3*a+1)/6,a*a*a/6];
 const output=new Uint8Array(width*height*4);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const u=Math.fround(Math.fround(Math.fround((x-scalars.cx)*Math.fround(scalars.scale))+scalars.cx)+scalars.tx),v=Math.fround(Math.fround(Math.fround((y-scalars.cy)*Math.fround(scalars.scale))+scalars.cy)+scalars.ty);let c;if(mode===0)c=at(Math.floor(u),Math.floor(v));else if(mode===1)c=linear(u,v);else{const px=Math.floor(u-.5),py=Math.floor(v-.5),wx=weights(u-.5-px,mode===4),wy=weights(v-.5-py,mode===4);c=0;for(let j=0;j<4;j++)for(let i=0;i<4;i++)c+=at(px+i-1,py+j-1)*wx[i]*wy[j];}const byte=Math.trunc(c*255)&255,p=(y*width+x)*4;output[p]=output[p+1]=output[p+2]=byte;}
 return output;
}
