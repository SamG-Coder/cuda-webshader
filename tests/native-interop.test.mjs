import test from 'node:test';
import assert from 'node:assert/strict';
import {supportsInteropRequirements,getNativeInteropCapabilities} from '../src/runtime/native-interop.js';
const capabilities={available:true,samePhysicalGpu:true,sharedBuffers:true,sharedTextures:true,
  textureFormats:['rgba8unorm','rgba16float','rgba32float','r32float'],gpuBufferToTexture:true,
  synchronization:'d3d12-fence-cuda-external-semaphore',maxResourceBytes:268435456,maxSharedBytes:2147483648,
  maxResources:256,maxBlocksPerLaunch:65536,maxTextureDimension2D:8192,bufferUsageMask:444,textureUsageMask:31,
  textureDimension:'2d',mipLevelCount:1,sampleCount:1,textureArrayLayers:1};
test('ClearWater resource requirements include both directions of GPU synchronization and allocation limits',()=>{
  const w=1920,h=1200,requirements={sharedBuffers:true,gpuBufferToTexture:true,resources:5,
    maxResourceBytes:w*h*16,sharedBytes:w*h*52+3*65536*16+5*65536,blocksPerLaunch:Math.ceil(w/8)*Math.ceil(h/8),bufferUsage:140};
  assert.equal(supportsInteropRequirements(capabilities,requirements),true);
  for(const patch of [{samePhysicalGpu:false},{sharedBuffers:false},{gpuBufferToTexture:false},
    {synchronization:'cpu-wait'},{maxResourceBytes:16*1024*1024},{maxSharedBytes:64*1024*1024},
    {maxResources:4},{maxBlocksPerLaunch:1024},{maxSharedBytes:undefined},{bufferUsageMask:12}])
    assert.equal(supportsInteropRequirements({...capabilities,...patch},requirements),false,JSON.stringify(patch));
  assert.equal(supportsInteropRequirements(capabilities,{blocksPerLaunch:129600}),false,'unsupported 4K grid must remain WebGPU');
});
test('texture sharing never implies arbitrary formats, arrays, mipmaps, multisampling or CPU mapping',()=>{
  assert.equal(supportsInteropRequirements(capabilities,{sharedTextures:true,textureFormats:['rgba8unorm','r32float'],textureUsage:31}),true);
  for(const requirement of [{textureFormats:['bgra8unorm']},{textureDimension:'3d'},
    {textureArrayLayers:2},{mipLevelCount:2},{sampleCount:4},{bufferUsage:1},{textureUsage:32}])
    assert.equal(supportsInteropRequirements(capabilities,requirement),false,JSON.stringify(requirement));
});
test('capability detection never requests permission and tolerates absent, denied or disconnected native APIs',async()=>{
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,'navigator');
  try {
    for(const cuda of [undefined,{queryPermission:async()=> 'denied',getInteropCapabilities(){throw Error('should not query');}},
      {queryPermission:async()=>{throw Error('disconnected');},getInteropCapabilities(){}}]) {
      Object.defineProperty(globalThis,'navigator',{configurable:true,value:{cuda}});
      const result=await getNativeInteropCapabilities({});assert.equal(result.available,false);assert.equal(result.sharedBuffers,false);
    }
  } finally {if(descriptor)Object.defineProperty(globalThis,'navigator',descriptor);else delete globalThis.navigator;}
});
