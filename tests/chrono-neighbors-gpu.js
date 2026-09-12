export async function checkChronoNeighbors(runtime){
 const source=await(await fetch(new URL('chrono-neighbors.cu',import.meta.url))).text();
 const cells=await runtime.kernel(source,{entry:'findCellStartEndD',workgroupSize:[128],sharedMemoryBytes:516,predicatedReturns:true});
 const mapping=await runtime.kernel(source,{entry:'OriginalToSortedD',workgroupSize:[128]});
 const native=await(await fetch(new URL('../reports/chrono-neighbors-native.json',import.meta.url))).json();
 const sizes=[0,1,31,127,128,129,255,256,257,1027];let compared=0;
 for(const n of sizes){
  const count=Math.max(1,n),hashes=Uint32Array.from({length:count},(_,i)=>Math.floor(i/7)),indices=Uint32Array.from({length:count},(_,i)=>n-1-i),cellCount=Math.ceil(count/7)+2;
  const reference=native.cases.find(c=>c.n===n);if(!reference)throw Error('Missing native Chrono case '+n);
  const expectedStart=new Uint32Array(cellCount),expectedEnd=new Uint32Array(cellCount);
  for(let i=0;i<n;i++){if(!i||hashes[i]!==hashes[i-1])expectedStart[hashes[i]]=i;expectedEnd[hashes[i]]=i+1;}
  const buffers={cellStartD:runtime.createBuffer(cellCount*4),cellEndD:runtime.createBuffer(cellCount*4),gridMarkerHashD:runtime.createBuffer(hashes),gridMarkerIndexD:runtime.createBuffer(indices)},out=runtime.createBuffer(new Uint32Array(count+4).fill(0xdeadbeef));
  try{
   runtime.batch().dispatch(cells.bind(buffers,{numActive:n}),[Math.max(1,Math.ceil(n/128))]).dispatch(mapping.bind({mapOriginalToSorted:out,gridMarkerIndex:buffers.gridMarkerIndexD},{numActive:n}),[Math.max(1,Math.ceil(n/128))]).submit();
   for(const [name,expected,captured] of [['cellStartD',expectedStart,reference.start],['cellEndD',expectedEnd,reference.end]]){const actual=await runtime.read(buffers[name],Uint32Array);if(captured.length!==actual.length)throw Error('Native cell length mismatch');for(let i=0;i<expected.length;i++){if(actual[i]!==expected[i]||actual[i]!==captured[i])throw Error(`Chrono ${name}, n=${n}, i=${i}: ${actual[i]} != ${expected[i]}`);compared++;}}
   const actual=await runtime.read(out,Uint32Array);if(reference.map.length!==actual.length)throw Error('Native map length mismatch');for(let i=0;i<actual.length;i++){const expected=i<n?n-1-i:0xdeadbeef;if(actual[i]!==expected||actual[i]!==reference.map[i])throw Error('Chrono mapping or inactive-lane write mismatch');compared++;}
  }finally{for(const buffer of [...Object.values(buffers),out])runtime.destroyBuffer(buffer);}
 }
 return {sizes,compared,nativeExact:true,partialBlocks:true,inactiveWrites:false,originalKernelBodies:true};
}
