export const transferCoordinates=[-.2,0,.0625,.13,.5,.875,1,1.2];
export const transferValues=[0,0,0,0,1,.25,0,1,0,1,.5,.75,.2,.4,1,.5];
export function transferSample(values,u,linear=true){const n=values.length/4,at=i=>values.slice(Math.max(0,Math.min(n-1,i))*4,Math.max(0,Math.min(n-1,i))*4+4);if(!linear)return at(Math.floor(u*n));const p=u*n-.5,i=Math.floor(p),a=at(i),b=at(i+1),t=p-i;return a.map((v,c)=>v*(1-t)+b[c]*t);}
