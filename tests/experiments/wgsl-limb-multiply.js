/*
 * The implementations contained in this file are heavily based on the
 * implementations found in the Berkeley SoftFloat library. As such, they are
 * licensed under the same 3-clause BSD license:
 *
 * License for Berkeley SoftFloat Release 3e
 *
 * John R. Hauser
 * 2018 January 20
 *
 * The following applies to the whole of SoftFloat Release 3e as well as to
 * each source file individually.
 *
 * Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 The Regents of the
 * University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions, and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions, and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the University nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Test-only fixed-limb multiplication, informed by Mesa/SoftFloat's wide-product approach.
// Reference: https://github.com/chaotic-cx/mesa-mirror/blob/7cda7850edd103ace21aac37d416d2fdf7a282e1/src/compiler/glsl/float64.glsl
// WGSL uses 16-bit limbs because GLSL umulExtended is unavailable here.
// Existing rounding, exponent handling and NaN policy remain unchanged.
import {FLOAT64_WGSL} from '../../src/compiler/float64.js';
const start=FLOAT64_WGSL.indexOf('  var product = vec4<u32>(0u);');
const original=FLOAT64_WGSL.slice(start,FLOAT64_WGSL.indexOf('\n}',start));
// Each schoolbook step is bounded by 65535^2 + 65535 + 65535 = 2^32-1.
// The 106-bit product shifts right by 49 or 50; discarded bits feed sticky.
const replacement=`  var p0=0u; var p1=0u; var p2=0u; var p3=0u; var p4=0u; var p5=0u; var p6=0u; var p7=0u;
  var carry0=0u;
  let t00=p0+((pa.significand.x >> 0u) & 65535u)*((pb.significand.x >> 0u) & 65535u)+carry0; p0=t00 & 65535u; carry0=t00 >> 16u;
  let t01=p1+((pa.significand.x >> 0u) & 65535u)*((pb.significand.x >> 16u) & 65535u)+carry0; p1=t01 & 65535u; carry0=t01 >> 16u;
  let t02=p2+((pa.significand.x >> 0u) & 65535u)*((pb.significand.y >> 0u) & 65535u)+carry0; p2=t02 & 65535u; carry0=t02 >> 16u;
  let t03=p3+((pa.significand.x >> 0u) & 65535u)*((pb.significand.y >> 16u) & 65535u)+carry0; p3=t03 & 65535u; carry0=t03 >> 16u;
  p4=carry0;
  var carry1=0u;
  let t10=p1+((pa.significand.x >> 16u) & 65535u)*((pb.significand.x >> 0u) & 65535u)+carry1; p1=t10 & 65535u; carry1=t10 >> 16u;
  let t11=p2+((pa.significand.x >> 16u) & 65535u)*((pb.significand.x >> 16u) & 65535u)+carry1; p2=t11 & 65535u; carry1=t11 >> 16u;
  let t12=p3+((pa.significand.x >> 16u) & 65535u)*((pb.significand.y >> 0u) & 65535u)+carry1; p3=t12 & 65535u; carry1=t12 >> 16u;
  let t13=p4+((pa.significand.x >> 16u) & 65535u)*((pb.significand.y >> 16u) & 65535u)+carry1; p4=t13 & 65535u; carry1=t13 >> 16u;
  p5=carry1;
  var carry2=0u;
  let t20=p2+((pa.significand.y >> 0u) & 65535u)*((pb.significand.x >> 0u) & 65535u)+carry2; p2=t20 & 65535u; carry2=t20 >> 16u;
  let t21=p3+((pa.significand.y >> 0u) & 65535u)*((pb.significand.x >> 16u) & 65535u)+carry2; p3=t21 & 65535u; carry2=t21 >> 16u;
  let t22=p4+((pa.significand.y >> 0u) & 65535u)*((pb.significand.y >> 0u) & 65535u)+carry2; p4=t22 & 65535u; carry2=t22 >> 16u;
  let t23=p5+((pa.significand.y >> 0u) & 65535u)*((pb.significand.y >> 16u) & 65535u)+carry2; p5=t23 & 65535u; carry2=t23 >> 16u;
  p6=carry2;
  var carry3=0u;
  let t30=p3+((pa.significand.y >> 16u) & 65535u)*((pb.significand.x >> 0u) & 65535u)+carry3; p3=t30 & 65535u; carry3=t30 >> 16u;
  let t31=p4+((pa.significand.y >> 16u) & 65535u)*((pb.significand.x >> 16u) & 65535u)+carry3; p4=t31 & 65535u; carry3=t31 >> 16u;
  let t32=p5+((pa.significand.y >> 16u) & 65535u)*((pb.significand.y >> 0u) & 65535u)+carry3; p5=t32 & 65535u; carry3=t32 >> 16u;
  let t33=p6+((pa.significand.y >> 16u) & 65535u)*((pb.significand.y >> 16u) & 65535u)+carry3; p6=t33 & 65535u; carry3=t33 >> 16u;
  p7=carry3;
  let product=vec4<u32>(p0 | (p1<<16u),p2 | (p3<<16u),p4 | (p5<<16u),p6 | (p7<<16u));
  let extra=select(0u,1u,(product.w & 512u)!=0u);
  let shift=17u+extra;
  let sticky=select(0u,1u,product.x!=0u || (product.y & ((1u<<shift)-1u))!=0u);
  let packed=vec2<u32>((product.y>>shift) | (product.z<<(32u-shift)) | sticky,(product.z>>shift) | (product.w<<(32u-shift)));
  return cw_d_pack(sign,pa.exponent+pb.exponent+i32(extra),packed);`;
export function optimizeLimbMultiply(artifact){
 if(!artifact.wgsl.includes(original))return {artifact,replacements:0};
 return {artifact:{...artifact,wgsl:artifact.wgsl.replace(original,replacement)},replacements:1};
}
