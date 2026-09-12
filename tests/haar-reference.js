// Independent pairwise Haar transform reference; test code only.
export function haarInitial(n){return Float32Array.from({length:n},(_,i)=>(i%29-14)*0.125);}
export function haarReference(input){const output=new Float32Array(input.length);let current=new Float32Array(input);while(current.length>1){const half=current.length/2,next=new Float32Array(half);for(let i=0;i<half;i++){next[i]=(current[2*i]+current[2*i+1])*Math.SQRT1_2;output[half+i]=(current[2*i]-current[2*i+1])*Math.SQRT1_2;}current=next;}output[0]=current[0];return output;}
