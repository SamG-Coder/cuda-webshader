// Bounded uncompressed 24-bit BMP loader. Retains file row order and zero alpha,
// matching CUDA sample image buffers. Use presentation flipY for bottom-up files.
export function decodeBMP(bytes){
 if(!(bytes instanceof Uint8Array)||bytes.length<54)throw Error('BMP requires a complete header.');const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),offset=v.getUint32(10,true),header=v.getUint32(14,true),width=v.getInt32(18,true),signedHeight=v.getInt32(22,true),height=Math.abs(signedHeight);
 if(v.getUint16(0,true)!==0x4d42||header<40||offset<14+header||offset>bytes.length||width<1||height<1||width>8192||height>8192||width*height>16777216||v.getUint16(26,true)!==1||v.getUint16(28,true)!==24||v.getUint32(30,true)!==0)throw Error('BMP requires bounded dimensions and uncompressed 24-bit RGB pixels.');
 const stride=Math.ceil(width*3/4)*4;if(offset+stride*height>bytes.length)throw Error('BMP pixel payload is truncated.');const data=new Float32Array(width*height*4);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const p=offset+y*stride+x*3,i=(y*width+x)*4;data[i]=bytes[p+2]/255;data[i+1]=bytes[p+1]/255;data[i+2]=bytes[p]/255;}
 return {width,height,bottomUp:signedHeight>0,data};
}
