export function convolutionTextureCase(scenario){
 const [width,height]=[[128,64],[35,19],[8,5],[512,512]][scenario],binomial=[1,16,120,560,1820,4368,8008,11440,12870,11440,8008,4368,1820,560,120,16,1];
 const coefficients=Array.from({length:17},(_,i)=>scenario===3?binomial[i]/65536:scenario===2?(i===8?1:0):(i+1)/256);
 return {width,height,coefficients,scalars:{imageW:width,imageH:height,...Object.fromEntries(coefficients.map((v,i)=>[`constant.c_Kernel[${i}]`,v]))}};
}
export function convolutionTextureInput(width,height){return Float32Array.from({length:width*height},(_,i)=>((i*37+Math.floor(i/width)*13)%256)/255);}
export function convolutionTextureReference(input,width,height,coefficients,vertical){
 const result=new Float32Array(input.length);for(let y=0;y<height;y++)for(let x=0;x<width;x++){let sum=0;for(let k=-8;k<=8;k++){const sx=vertical?x:Math.max(0,Math.min(width-1,x+k)),sy=vertical?Math.max(0,Math.min(height-1,y+k)):y;sum+=input[sy*width+sx]*coefficients[8-k];}result[y*width+x]=sum;}return result;
}
