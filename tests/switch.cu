// MIT control-flow fixture, compiled unchanged by native CUDA and WebShader.
enum class Choice { FIRST = -2, SECOND = 1, LAST = 3 };
__device__ int choose(int value) {
 int result=0;
 switch(value++) {
 case -2: result+=2;
 case -1: case 0: result+=3; if(value==0)break;
 default: result+=5;
 case 3: { int extra=7; result+=extra; break; }
 case 4: return 41;
 }
 return result*10+value;
}
__device__ int loops(int value){
 int result=0;
 for(int i=0;i<5;i++){
  switch((value+i)%4){
  case 0: continue;
  case 1: for(int j=0;j<3;j++){if(j==2)break;result++;} break;
  case 2: switch(value){case -2: result+=20;break;default:result+=4;} break;
  default: result+=8;
  }
  result+=100;
 }
 return result;
}
__global__ void switchCases(const int* input,int* output){
 unsigned int i=threadIdx.x;int value=input[i];
 output[4*i]=choose(value);output[4*i+1]=loops(value);
 int result=19;
 switch(value){case -2:{result=2;break;}case 1:{result=4;break;}case 3:{result=8;break;}}
 output[4*i+2]=result;
 unsigned int bits=(unsigned int)(value);
 switch(bits){case 4294967295u:result=31;break;case 0u:result=17;break;default:result=9;}
 output[4*i+3]=result;
}
