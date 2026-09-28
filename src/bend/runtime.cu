// CUDA WebShader Bend term machine. Each invocation owns an isolated arena.
// Values, environments and continuations use explicit 32-bit buffer offsets.
// The checked frontend owns code generation; no arbitrary program buffers are accepted.
__device__ unsigned int bend_alloc(unsigned int* heap, unsigned int base,
  unsigned int& hp, unsigned int kind, unsigned int a, unsigned int b, unsigned int c) {
  unsigned int p=hp; hp+=4u;
  heap[base+p]=kind; heap[base+p+1u]=a; heap[base+p+2u]=b; heap[base+p+3u]=c;
  return p;
}

__global__ void bend_run(const unsigned int* inputs, unsigned int* outputs,
  unsigned int* status, unsigned int* usage, unsigned int* heap,
  unsigned int count, unsigned int arena_words, unsigned int max_steps) {
  unsigned int lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=count) return;
  unsigned int base=lane*arena_words, hp=4u, pc=BEND_ROOT, env=0u, cont=0u;
  unsigned int value=0u, mode=0u, error=0u, steps=0u;
  // mode 0 evaluates a term; mode 1 returns a value to a continuation;
  // mode 2 applies a closure to an already evaluated argument.
  unsigned int function_value=0u, argument=0u;
  for(steps=0u;steps<max_steps;steps++) {
    // A transition allocates at most 12 four-word cells (8 constructor fields).
    if(hp+64u>arena_words) {error=1u;break;}
    if(mode==0u) {
      unsigned int op=bend_code[pc*4u], a=bend_code[pc*4u+1u];
      unsigned int b=bend_code[pc*4u+2u];
      if(op==1u) {value=bend_alloc(heap,base,hp,a,b,0u,0u);mode=1u;}
      else if(op==2u) {
        unsigned int e=env, found=0u;
        while(e!=0u) {if(heap[base+e+1u]==a){value=heap[base+e+2u];found=1u;break;}e=heap[base+e+3u];}
        if(found==0u){error=3u;break;}mode=1u;
      }
      else if(op==3u || op==6u) {value=bend_alloc(heap,base,hp,op==3u?5u:6u,pc,env,0u);mode=1u;}
      else if(op==4u) {cont=bend_alloc(heap,base,hp,20u,b,env,cont);pc=a;}
      else if(op==5u) {pc=a;env=0u;}
      else if(op==7u) {value=bend_alloc(heap,base,hp,7u,a,b,0u);mode=1u;}
      else if(op==8u) {
        if(b==0u) value=bend_alloc(heap,base,hp,a==1u?2u:4u,a==1u?0u:a,0u,0u);
        else value=bend_alloc(heap,base,hp,8u,a,b,0u);
        mode=1u;
      }
      else if(op==9u) {value=bend_alloc(heap,base,hp,a,inputs[lane*BEND_ARGS+b],0u,0u);mode=1u;}
      else {error=4u;break;}
    } else if(mode==1u) {
      if(cont==0u) break;
      unsigned int k=heap[base+cont], a=heap[base+cont+1u], b=heap[base+cont+2u], next=heap[base+cont+3u];
      if(k==20u) {cont=bend_alloc(heap,base,hp,21u,value,0u,next);pc=a;env=b;mode=0u;}
      else if(k==21u) {function_value=a;argument=value;cont=next;mode=2u;}
      else if(k==22u) {function_value=value;argument=a;cont=next;mode=2u;}
      else {error=3u;break;}
    } else {
      unsigned int kind=heap[base+function_value], a=heap[base+function_value+1u];
      unsigned int b=heap[base+function_value+2u], c=heap[base+function_value+3u];
      if(kind==5u) {
        env=bend_alloc(heap,base,hp,9u,bend_code[a*4u+1u],argument,b);
        pc=bend_code[a*4u+2u];mode=0u;
      } else if(kind==6u) {
        unsigned int vk=heap[base+argument], va=heap[base+argument+1u];
        unsigned int match_tag=vk==2u?(va==0u?1u:2u):va;
        unsigned int want=bend_code[a*4u+1u];
        env=b;
        if(match_tag==want) {
          if(vk==2u && va!=0u) {
            unsigned int pred=bend_alloc(heap,base,hp,2u,va-1u,0u,0u);
            cont=bend_alloc(heap,base,hp,22u,pred,0u,cont);
          } else if(vk==4u) {
            unsigned int fields=heap[base+argument+2u];
            while(fields!=0u) {
              cont=bend_alloc(heap,base,hp,22u,heap[base+fields+1u],0u,cont);
              fields=heap[base+fields+2u];
            }
          } else if(vk!=2u) {error=3u;break;}
          pc=bend_code[a*4u+2u];
        } else {
          cont=bend_alloc(heap,base,hp,22u,argument,0u,cont);pc=bend_code[a*4u+3u];
        }
        mode=0u;
      } else if(kind==7u || kind==8u) {
        if(b>1u) {
          unsigned int field=bend_alloc(heap,base,hp,10u,argument,c,0u);
          value=bend_alloc(heap,base,hp,kind,a,b-1u,field);mode=1u;
        } else if(kind==8u) {
          if(a==2u) {
            unsigned int n=heap[base+argument+1u];
            if(n==4294967295u){error=5u;break;}
            value=bend_alloc(heap,base,hp,2u,n+1u,0u,0u);
          } else {
            unsigned int field=bend_alloc(heap,base,hp,10u,argument,c,0u);
            value=bend_alloc(heap,base,hp,4u,a,field,0u);
          }
          mode=1u;
        } else {
          unsigned int x=heap[base+argument+1u], y=x;
          if(c!=0u) {unsigned int first=heap[base+c+1u];x=heap[base+first+1u];}
          unsigned int r=0u, rk=1u;
          if(a==1u) r=x+y;
          else if(a==2u) r=x-y;
          else if(a==3u) r=x*y;
          else if(a==4u) r=y==0u?0u:x/y;
          else if(a==5u) r=y==0u?x:x%y;
          else if(a==6u) {r=x==y?4u:3u;rk=4u;}
          else if(a==7u) {r=x<y?4u:3u;rk=4u;}
          else if(a==8u) r=x&y;
          else if(a==9u) r=x|y;
          else if(a==10u) r=x^y;
          else if(a==11u) r=x<<1u;
          else if(a==12u) r=x>>1u;
          else if(a==24u) r=y>=32u?0u:x<<y;
          else if(a==25u) r=y>=32u?0u:x>>y;
          else {
            float fx=__uint_as_float(x), fy=__uint_as_float(y), fr=0.0f;rk=3u;
            if(a==13u) fr=fx+fy;
            else if(a==14u) fr=fx-fy;
            else if(a==15u) fr=fx*fy;
            else if(a==16u) fr=fx/fy;
            else if(a==17u) {r=fx==fy?4u:3u;rk=4u;}
            else if(a==18u) {r=fx<fy?4u:3u;rk=4u;}
            else if(a==19u) fr=-fx;
            else if(a==20u) fr=fabsf(fx);
            else if(a==21u) fr=sqrtf(fx);
            else if(a==22u) fr=sinf(fx);
            else if(a==23u) fr=cosf(fx);
            if(rk==3u) r=__float_as_uint(fr);
          }
          value=bend_alloc(heap,base,hp,rk,r,0u,0u);mode=1u;
        }
      } else {error=3u;break;}
    }
  }
  if(error==0u && steps>=max_steps) error=2u;
  if(error==0u && heap[base+value]!=BEND_RESULT) error=6u;
  outputs[lane]=error==0u?heap[base+value+1u]:0u;
  status[lane]=error;usage[lane*2u]=hp;usage[lane*2u+1u]=steps;
}
