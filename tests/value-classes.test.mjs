import {readFileSync} from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=`class vec3 {
public:
 __host__ __device__ vec3() {}
 __host__ __device__ vec3(float e0,float e1,float e2){e[0]=e0;e[1]=e1;e[2]=e2;}
 __host__ __device__ inline float x() const{return e[0];}
 __host__ __device__ inline float y() const{return e[1];}
 __host__ __device__ inline float z() const{return e[2];}
 __host__ __device__ inline float squared_length() const{return e[0]*e[0]+e[1]*e[1]+e[2]*e[2];}
 float e[3];
};`;
test('Value-class constructors and const methods preserve array fields and copies',()=>{
 const c=compile(source+'__global__ void k(float*out){vec3 a(2.0f,3.0f,4.0f);vec3 b=a;a.e[0]=9.0f;vec3 c=vec3(1.0f,2.0f,3.0f);out[0]=b.x();out[1]=b.y();out[2]=b.z();out[3]=b.squared_length();out[4]=c.squared_length();}',{workgroupSize:[1,1,1]}),out=new Float32Array(5);
 executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[2,3,4,29,14]);
});
test('Value classes reject unsupported access and const mutation',()=>{
 for(const body of ['class V {float x;};','class V {public: float x;__device__ float get() const{x=2.0f;return x;}};'])assert.throws(()=>compile(body+'__global__ void k(float*out){V v;out[0]=v.get();}'));
});
test('Resolved value-class method recursion is rejected',()=>{
 assert.throws(()=>compile('class V{public:float e;__device__ V(float x){e=x;}__device__ float get() const{V other(1.0f);return other.get();}};__global__ void k(float*out){V v(1.0f);out[0]=v.get();}'),/Recursive/);
});
const operators=source.replace(' float e[3];',` __device__ const vec3& operator+() const{return *this;}
 __device__ vec3 operator-() const{return vec3(-e[0],-e[1],-e[2]);}
 __device__ float operator[](int i) const{return e[i];}
 float e[3];`);
test('Const self, unary negation and indexing preserve class operator values',()=>{
 const c=compile(operators+'__global__ void k(float*out){vec3 v(2.0f,-3.0f,4.0f);vec3 a=+v;vec3 b=-v;out[0]=a.x();out[1]=b.y();out[2]=v[2];out[3]=(-v)[1];}',{workgroupSize:[1,1,1]}),out=new Float32Array(4);
 executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[2,3,4,3]);
 for(const statement of ['v[0]=1.0f;','(+v).e[0]=1.0f;'])assert.throws(()=>compile(operators+'__global__ void k(){vec3 v(1.0f,2.0f,3.0f);'+statement+'}'),/Read-only/);
});

const writableOperators=operators.replace(' float e[3];',' __device__ float& operator[](int i){return e[i];} float e[3];');
test('Writable class index aliases original array storage with dynamic indices',()=>{
 const c=compile(writableOperators+'__global__ void k(float*out){vec3 v(2.0f,3.0f,4.0f);int i=0;v[i]=9.0f;v[1]+=5.0f;out[0]=v.x();out[1]=v.y();out[2]=v[2];out[3]=float(i);}',{workgroupSize:[1,1,1]}),out=new Float32Array(4);
 executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[9,8,4,0]);
 assert.throws(()=>compile(writableOperators+'__global__ void k(){const vec3 v(1.0f,2.0f,3.0f);v[0]=1.0f;}'),/const/);
 assert.throws(()=>compile(writableOperators.replace('return e[i];} float e[3];','return e[0];} float e[3];')+'__global__ void k(){}'),/Reference indexing/);
});

const mutable=source.replace(' float e[3];',` __device__ vec3& operator+=(const vec3 &other);
 __device__ void clear();
 float e[3];`)+`
 __device__ vec3& vec3::operator+=(const vec3 &v){e[0]+=v.e[0];e[1]+=v.e[1];e[2]+=v.e[2];return *this;}
 __device__ void vec3::clear(){e[0]=0.0f;e[1]=0.0f;e[2]=0.0f;}`;
test('External mutable methods and compound class operators update the receiver',()=>{
 const c=compile(mutable+'__global__ void k(float*out){vec3 a(1.0f,2.0f,3.0f),b(4.0f,5.0f,6.0f);a+=b;out[0]=a.x();out[1]=a.y();out[2]=a.z();b.clear();out[3]=b.squared_length();}',{workgroupSize:[1,1,1]}),out=new Float32Array(4);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[5,7,9,0]);
 assert.throws(()=>compile(mutable+'__global__ void k(){const vec3 a(1.0f,2.0f,3.0f);a.clear();}'),/mutable/);
 assert.throws(()=>compile(mutable+'__global__ void k(){vec3 a(1.0f,2.0f,3.0f);a+=a;}'),/Aliased/);
 assert.throws(()=>compile(mutable.replace('vec3::clear()', 'vec3::missing()')+'__global__ void k(){}'),/signature/);
 assert.throws(()=>compile(mutable.replace('return *this;','return v;')+'__global__ void k(){}'),/return/);
});

test('Original device vec3 definitions support free operators and readonly self dot products',()=>{
 const full=readFileSync(new URL('./pathtracer-value-class.cuh',import.meta.url),'utf8');
 const c=compile(full+'__global__ void k(float*out){vec3 a(2.0f,4.0f,8.0f),b(1.0f,2.0f,4.0f);vec3 sum=a+b,delta=a-b,product=a*b,ratio=a/b,scaled=2.0f*a,other=a*2.0f,half=a/2.0f;out[0]=sum.x();out[1]=delta.y();out[2]=product.z();out[3]=ratio.x();out[4]=scaled.y();out[5]=other.z();out[6]=half.x();out[7]=dot(a,a);vec3 normal=unit_vector(vec3(3.0f,0.0f,4.0f));out[8]=normal.x();}',{workgroupSize:[1,1,1]}),out=new Float32Array(9);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[3,2,32,2,8,16,1,84,Math.fround(0.6)]);
 assert.throws(()=>compile(source+'__device__ vec3 operator+(vec3 a){return a;}__global__ void k(){}'),/two/);
 assert.throws(()=>compile(source+'__device__ vec3 operator+(vec3 &a,vec3 b){return b;}__global__ void k(){}'),/const-reference/);
});

test('Original ray stores independent nested vectors and evaluates ray points',()=>{
 const full=readFileSync(new URL('./pathtracer-value-class.cuh',import.meta.url),'utf8');const c=compile(full+'__global__ void k(float*out){vec3 a(1.0f,2.0f,3.0f),b(2.0f,4.0f,8.0f);ray r(a,b),copy=r;r.A.e[0]=-99.0f;vec3 point=copy.point_at_parameter(2.0f);out[0]=point.x();out[1]=point.y();out[2]=point.z();}',{workgroupSize:[1,1,1]}),out=new Float32Array(3);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[5,10,19]);
 assert.throws(()=>compile('class R{public:R self;};__global__ void k(){}'),/previously completed/);
});
test('Nested class default constructors execute before the enclosing constructor body',()=>{
 const c=compile('class Inner{public:int x;__device__ Inner(){x=7;}};class Outer{public:Inner v;__device__ Outer(){v.x+=2;}};__global__ void k(int*out){Outer o;out[0]=o.v.x;}',{workgroupSize:[1,1,1]}),out=new Int32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],9);
 const implicit=compile('class Inner{public:int x;__device__ Inner(){x=7;}};class Outer{public:Inner v;};__global__ void k(int*out){Outer o;out[0]=o.v.x;}',{workgroupSize:[1,1,1]});executeCPU(implicit,{out},{},[1]);assert.equal(out[0],7);
});

test('Member initializer lists follow field order and initialize nested values without default constructors',()=>{
 const src='class Inner{public:int x;__device__ Inner(int v):x(v){}};class Outer{public:int first;int second;Inner nested;__device__ Outer():nested(9),second(first+2),first(5){}};__global__ void k(int*out){Outer o;out[0]=o.first;out[1]=o.second;out[2]=o.nested.x;}';
 const c=compile(src,{workgroupSize:[1,1,1]}),out=new Int32Array(3);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[5,7,9]);
 assert.throws(()=>compile(src.replace('first(5)','first(5),first(6)')),/Duplicate member/);
 assert.throws(()=>compile(src.replace('first(5)','missing(5)')),/Unknown member/);
});

test('Original sphere intersection preserves hit records across near, far, tangent and missed rays',()=>{
 const full=readFileSync(new URL('./pathtracer-value-class.cuh',import.meta.url),'utf8');const c=compile(full+'__global__ void k(float*out){int i=threadIdx.x;vec3 center(0.0f,0.0f,-1.0f),origin(i==1?2.0f:(i==4?0.5f:0.0f),0.0f,i==2?-1.0f:0.0f),direction(0.0f,0.0f,-1.0f);sphere s(center,0.5f,NULL);ray r(origin,direction);hit_record rec;rec.t=-1.0f;rec.mat_ptr=nullptr;bool ok=s.hit(r,i==3?0.75f:0.0f,100.0f,rec);out[i*3]=float(ok);out[i*3+1]=rec.t;out[i*3+2]=float(rec.mat_ptr==NULL);}',{workgroupSize:[5,1,1]}),out=new Float32Array(15);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[1,0.5,1,0,-1,1,1,0.5,1,1,1.5,1,0,-1,1]);
});
test('Object reference stage rejects fabricated addresses, arithmetic and unimplemented interfaces',()=>{
 for(const stmt of ['M*p=1;','M*p=NULL;p=p+1;','M*p=NULL;float x=p->x;'])assert.throws(()=>compile('class M;__global__ void k(){'+stmt+'}'));
 assert.throws(()=>compile('class Base{public:__device__ virtual bool hit() const=0;};class D:public Base{public:int x;};__global__ void k(){}'),/implement/);
});

test('Concrete new objects preserve identity, member access and released slot reuse',()=>{
 const c=compile('class V{public:float x;__device__ V(float a):x(a){} __device__ float get() const{return x;}__device__ void set(float v){x=v;}};__global__ void k(float*out){V*p=new V(3.0f);V*q=new V(7.0f);out[0]=p->get();out[1]=q->x;out[2]=float(p!=q);delete p;V*r=new V(10.0f);r->set(11.0f);out[3]=r->get();out[4]=q->get();delete r;delete q;}',{workgroupSize:[1,1,1]}),out=new Float32Array(5);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[3,7,1,11,7]);assert.equal(c.metadata.objectHeap.scope,'invocation');
});

test('Tagged base pointers select distinct implementations at runtime',()=>{
 const fixture=readFileSync(new URL('./pathtracer-dispatch-fixture.cuh',import.meta.url),'utf8');const c=compile(fixture+'__global__ void k(float*out){DispatchBase*a=new DispatchScale(2.0f);DispatchBase*b=new DispatchOffset(7.0f);DispatchBase*p=a;if(threadIdx.x%2)p=b;out[threadIdx.x]=p->value(3.0f);delete a;delete b;}',{workgroupSize:[4,1,1]}),out=new Float32Array(4);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[6,10,6,10]);assert.notEqual(c.metadata.objectHeap.types[0].tag,c.metadata.objectHeap.types[1].tag);
});

test('Persistent object pointer buffers require explicit arena mode and storage layout metadata',()=>{
 const source='class V{public:float x;__device__ V(float v):x(v){}};__global__ void create(V**out){out[0]=new V(3.0f);}';
 assert.throws(()=>compile(source),/require objectHeap/);
 const c=compile(source,{objectHeap:'persistent'});assert.equal(c.metadata.bindings[0].stride,4);assert.equal(c.metadata.objectHeap.persistent,true);assert.equal(c.metadata.objectHeap.types[0].byteLength,8192);assert.match(c.wgsl,/atomicCompareExchangeWeak/);assert.match(c.wgsl,/@group\(1\)/);assert.throws(()=>executeCPU(c,{out:new Uint32Array(1)},{},[1]),/no cross-dispatch arena/);
});


test('Original hitable_list captures persistent buffer origins across entries',()=>{
 const text=['value-class','list-class','list-kernels'].map(n=>readFileSync(new URL('./pathtracer-'+n+'.cuh',import.meta.url),'utf8')).join('\n');
 const artifacts=['create_list','trace_list','update_list','free_list'].map(entry=>compile(text,{entry,objectHeap:'persistent',workgroupSize:[1,1,1]}));
 for(const a of artifacts){assert.deepEqual(a.metadata.objectHeap.imports.map(i=>[i.name,i.targets]),[['objects',['sphere']],['world',['hitable_list']]]);assert.deepEqual(a.metadata.objectHeap,artifacts[0].metadata.objectHeap);}
 assert.throws(()=>compile(text,{entry:'trace_list'}),/persistent object arenas/);
});


test('Grouped class fields and mutable value returns preserve receiver updates',()=>{
 const s='class V{public:int x,y;__device__ V(int a){x=a;y=2;}__device__ int next(){x+=y;return x;}};__global__ void k(int*out){V v(1);out[0]=v.next();out[1]=v.next();out[2]=v.x;}';
 const a=compile(s,{workgroupSize:[1,1,1]}),out=new Int32Array(3);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[3,5,5]);
 assert.throws(()=>compile(s.replace('int x,y','int x,x')),/duplicate/);
 assert.throws(()=>compile(s.replace('V v(1)','const V v(1)')),/mutable/);
});

test('Local record pointers preserve state through class methods and forwarded helpers',()=>{
 const s='struct S{unsigned int x;};__device__ unsigned int next(S*s){s->x+=3;return s->x;}__device__ unsigned int forward(S*s){return next(s);}class V{public:int x;__device__ V(){x=1;}__device__ unsigned int get(S*s){return forward(s);}};__global__ void k(unsigned int*out){S s;s.x=1;V v;out[0]=v.get(&s);out[1]=s.x;}';
 const a=compile(s,{workgroupSize:[1,1,1]}),out=new Uint32Array(2);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[4,4]);
 assert.throws(()=>compile(s.replace('return next(s)','return next(s+1)')),/Local pointers/);
});

test('XORWOW compatibility is explicit and rejects unsupported initialization modes',()=>{
 const s='__global__ void k(unsigned int*out){curandState state;curand_init(1984u,0,0,&state);out[0]=curand(&state);}';
 const options={libraries:['curand-xorwow'],workgroupSize:[1,1,1]};
 const a=compile(s,options);assert.equal(a.metadata.libraries[0].seedBits,64);
 assert.throws(()=>compile(s));
 for(const replacement of ['1984u,1,0','1984u,0,1','1984u,threadIdx.x,0','1.5f,0,0'])assert.throws(()=>compile(s.replace('1984u,0,0',replacement),options),/XORWOW/);
 assert.throws(()=>compile(s,{libraries:['unknown']}),/Supported libraries/);
 const camera=['value-class','camera-class','camera-kernels'].map(n=>readFileSync(new URL('./pathtracer-'+n+'.cuh',import.meta.url),'utf8')).join('\n');assert.ok(compile(camera,{...options,entry:'check_camera'}).wgsl.includes('tan('));
});


test('Object call macros evaluate their body at each use and retain state effects',()=>{
 const s='#define NEXT next(&state)\n__device__ float next(int*p){*p+=1;return float(*p);}__global__ void k(float*out){int state=0;out[0]=NEXT;out[1]=NEXT;out[2]=float(state);}';
 const a=compile(s,{workgroupSize:[1,1,1]}),out=new Float32Array(3);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[1,2,2]);
 assert.throws(()=>compile(s.replace('next(&state)','next(&state);next(&state)')),/bounded|Template|Object/);
});

test('Original material methods compile with call macros, float pow and double constructor literals',()=>{
 const source=['class','kernels'].map(n=>readFileSync(new URL('./pathtracer-material-'+n+'.cuh',import.meta.url),'utf8')).join('\n');
 const a=compile(source,{entry:'check_material',libraries:['curand-xorwow'],workgroupSize:[1,1,1],objectHeap:'persistent'});assert.equal(a.metadata.objectHeap.types.length,3);assert.ok(a.wgsl.includes('pow('));
 const c=compile('class V{public:float x;__device__ V(){x=0.0f;}__device__ V(float a){x=a;}};__global__ void k(float*out){V a(0.7);out[0]=a.x;out[1]=pow(0.5f,3.0f);}',{workgroupSize:[1,1,1]}),out=new Float32Array(2);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[Math.fround(0.7),0.125]);
});


test('Explicit related class downcasts retain allocation identity and concrete fields',()=>{
 const source='class B{public:__device__ virtual int get() const=0;};class D:public B{public:int x;__device__ D(){x=42;}__device__ int get()const{return x;}};__global__ void k(int*out){B*b=new D();D*d=(D*)b;out[0]=d->x;out[1]=d->get();delete b;}';
 const a=compile(source,{workgroupSize:[1,1,1]}),out=new Int32Array(2);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[42,42]);
 assert.throws(()=>compile(source.replace('D*d=(D*)b','D*d=b')),/Cannot convert/);
 assert.throws(()=>compile('class X;__global__ void k(){X*p=new X();}'),/complete concrete/);
});

test('Record buffers expose aligned storage strides and reject non-shareable fields',()=>{
 const a=compile('struct R{unsigned int d;unsigned int v[5];};__global__ void k(R*out){out[0].v[4]=9u;}');assert.equal(a.metadata.bindings[0].stride,24);
 const b=compile('class V{public:float e[3];};__global__ void k(V*out){out[0].e[2]=1.0f;}',{valueBuffers:['out']});assert.equal(b.metadata.bindings[0].stride,12);
 assert.throws(()=>compile('struct R{bool flag;};__global__ void k(R*out){out[0].flag=true;}'),/host-shareable/);
 assert.throws(()=>compile('class V{public:float x;};__global__ void k(V*out){}',{valueBuffers:['missing']}),/Unknown value buffer/);
});

test('Wide integer seed shifts preserve both words and signed conversion',()=>{
 const a=compile('__global__ void k(unsigned int*out){size_t n=(size_t)(-1);out[0]=(unsigned int)(n>>32u);out[1]=(unsigned int)(n<<32u);}',{workgroupSize:[1,1,1]}),out=new Uint32Array(2);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[4294967295,0]);
 assert.throws(()=>compile('__global__ void k(unsigned int*out){size_t n=(size_t)1;out[0]=(unsigned int)(n>>64u);}'),/0..63/);
});

test('Private state remains accessible through class methods and same-class receivers',()=>{
 const s=readFileSync('tests/private-class.cu','utf8'),a=compile(s,{entry:'private_class_values',workgroupSize:[32]}),out=new Int32Array(32);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],Array.from({length:32},(_,i)=>3*i+7));
 for(const expression of ['a.value','a.twice()'])assert.throws(()=>compile(s+'__global__ void invalid(int* out){Counter a(1);out[0]='+expression+';} ',{entry:'invalid'}),/Private/);
 assert.throws(()=>compile('class Hidden {__device__ Hidden(){}public:int value;};__global__ void k(){Hidden h;}',{entry:'k'}),/Private/);
});

test('Public index references can expose private arrays without exposing their fields',()=>{
 const s='class Values {float data[2];public:__device__ Values(){data[0]=1.f;data[1]=2.f;}__device__ float& operator[](int i){return data[i];}};__global__ void k(float* out){Values v;v[1]=7.f;out[0]=v[1];}';
 const a=compile(s,{entry:'k',workgroupSize:[1]}),out=new Float32Array(1);executeCPU(a,{out},{},[1]);assert.equal(out[0],7);
 assert.throws(()=>compile(s.replace('out[0]=v[1]','out[0]=v.data[1]'),{entry:'k'}),/Private/);
 assert.throws(()=>compile(s.replace('public:','public:').replace('__device__ float& operator[]','private:__device__ float& operator[]'),{entry:'k'}),/Private/);
});

test('Original nested field getters preserve const references after owner mutation',()=>{
 const s=readFileSync('tests/quadtree-field-references.cu','utf8'),a=compile(s,{entry:'field_references',workgroupSize:[32]}),out=new Float32Array(96);executeCPU(a,{out},{},[1]);for(let i=0;i<32;i++)assert.deepEqual([...out.slice(i*3,i*3+3)],[1,i+2,i+3]);
 assert.throws(()=>compile(s.replace('out[i*3]=corner.x;','corner.x=5.f;'),{entry:'field_references'}),/const/);
 assert.throws(()=>compile(s.replace('const float2 &corner','float2 &corner'),{entry:'field_references'}),/const/);
 assert.throws(()=>compile(s.replace('return m_p_max;','return make_float2(0.f,0.f);'),{entry:'field_references'}),/exactly return field/);
});

test('Original Parameters constructors initialize const fields and retain the parent values',()=>{
 const s=readFileSync('tests/quadtree-parameters.cu','utf8'),a=compile(s,{entry:'parameter_values',workgroupSize:[32]}),out=new Int32Array(192);executeCPU(a,{out},{},[1]);for(let i=0;i<32;i++)assert.deepEqual([...out.slice(i*6,i*6+6)],[0,1,0,16,i+5,16]);
 for(const statement of ['root.max_depth=9;','root=child;','grandchild.min_points_per_node++;'])assert.throws(()=>compile(s.replace('out[i*6]=root.depth;',statement),{entry:'parameter_values'}),/const/);
 assert.throws(()=>compile(s.replace(', max_depth(max_depth)',''),{entry:'parameter_values'}),/initializer/);
});

test('Original quadtree storage references bind whole nodes and nested mutable methods',()=>{const s=readFileSync(new URL('quadtree-storage-references.cu',import.meta.url),'utf8'),options={valueBuffers:['nodes'],workgroupSize:[32]},a=compile(s,{...options,entry:'write_nodes'}),b=compile(s,{...options,entry:'read_nodes'});assert.equal(a.metadata.bindings.find(b=>b.name==='nodes').readOnly,false);assert.equal(b.metadata.bindings.find(b=>b.name==='nodes').readOnly,true);assert.throws(()=>compile(s.replace('Quadtree_node &node=nodes[i]','const Quadtree_node &node=nodes[i]'),{...options,entry:'write_nodes'}),/const|mutable/);});
