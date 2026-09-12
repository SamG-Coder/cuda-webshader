// Decode an 8-bit binary PGM input image; pixel transforms remain on the GPU.
export function decodePGM(bytes){
 if(!(bytes instanceof Uint8Array))throw Error('PGM input must be bytes.');let offset=0;
 const whitespace=b=>[9,10,11,12,13,32].includes(b);
 const token=()=>{while(offset<bytes.length){if(whitespace(bytes[offset]))offset++;else if(bytes[offset]===35){while(offset<bytes.length&&bytes[offset]!==10)offset++;}else break;}const start=offset;while(offset<bytes.length&&!whitespace(bytes[offset])&&bytes[offset]!==35)offset++;return new TextDecoder().decode(bytes.subarray(start,offset));};
 if(token()!=='P5')throw Error('Only binary P5 PGM images are supported.');
 const width=Number(token()),height=Number(token()),max=Number(token());
 if(![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=8192)||width*height>16777216||max!==255||!whitespace(bytes[offset]))throw Error('PGM requires bounded dimensions and 8-bit pixels (max 255).');
 if(bytes[offset]===13&&bytes[offset+1]===10)offset+=2;else offset++;
 if(bytes.length-offset!==width*height)throw Error('PGM pixel byte count does not match dimensions.');
 return {width,height,data:Float32Array.from(bytes.subarray(offset),v=>v/255)};
}
