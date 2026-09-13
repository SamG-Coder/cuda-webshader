export async function checkSwitch(runtime){
 const source=await(await fetch(new URL('switch.cu',import.meta.url))).text(),expected=await(await fetch(new URL('../reports/switch-native.json',import.meta.url))).json();
 const kernel=await runtime.kernel(source,{entry:'switchCases',workgroupSize:[9]}),input=runtime.createBuffer(new Int32Array([-3,-2,-1,0,1,2,3,4,5])),output=runtime.createBuffer(36*4);
 try{runtime.batch().dispatch(kernel.bind({input,output},{}),[1]).submit();const actual=await runtime.read(output,Int32Array);if(JSON.stringify([...actual])!==JSON.stringify(expected))throw Error('Switch control flow differs from native CUDA');return {nativeExact:true,compared:36,fallThrough:true,nestedBreakAndContinue:true,selectorEvaluatedOnce:true};}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
}
