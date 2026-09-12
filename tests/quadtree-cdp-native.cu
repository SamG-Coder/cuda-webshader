// Original NVIDIA quadtree kernels are included unchanged. This harness captures
// their complete default workload for an independent browser comparison.
#define main nvidia_quadtree_sample_main
#include "../.local/nvidia-audit/cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu"
#undef main
#include <fstream>
#include <vector>
#include <functional>
#include <algorithm>
#include <array>
static void save(const char* path,const void* p,size_t bytes){std::ofstream f(path,std::ios::binary);f.write((const char*)p,bytes);if(!f)exit(2);}
int main(){
 const int n=1024,maxDepth=8,minPoints=16,capacity=21845;
 thrust::device_vector<float> x0(n),y0(n),x1(n),y1(n);
 Random_generator rng;thrust::generate(thrust::make_zip_iterator(x0.begin(),y0.begin()),thrust::make_zip_iterator(x0.end(),y0.end()),rng);
 thrust::host_vector<float> originalX(x0),originalY(y0);std::vector<float> input(2*n);for(int i=0;i<n;i++){input[2*i]=originalX[i];input[2*i+1]=originalY[i];}save("reports/quadtree-input.bin",input.data(),input.size()*4);
 Points descriptors[2];descriptors[0].set(thrust::raw_pointer_cast(x0.data()),thrust::raw_pointer_cast(y0.data()));descriptors[1].set(thrust::raw_pointer_cast(x1.data()),thrust::raw_pointer_cast(y1.data()));
 Points* points;Quadtree_node* nodes;checkCudaErrors(cudaMalloc(&points,sizeof(descriptors)));checkCudaErrors(cudaMemcpy(points,descriptors,sizeof(descriptors),cudaMemcpyHostToDevice));checkCudaErrors(cudaMalloc(&nodes,capacity*sizeof(Quadtree_node)));checkCudaErrors(cudaMemset(nodes,0,capacity*sizeof(Quadtree_node)));
 Quadtree_node root;root.set_range(0,n);checkCudaErrors(cudaMemcpy(nodes,&root,sizeof(root),cudaMemcpyHostToDevice));Parameters params(maxDepth,minPoints);
 build_quadtree_kernel<128><<<1,128,64>>>(nodes,points,params);checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaDeviceSynchronize());
 thrust::host_vector<float> outputX(x0),outputY(y0);Points hostPoints;hostPoints.set(outputX.data(),outputY.data());std::vector<Quadtree_node> hostNodes(capacity);checkCudaErrors(cudaMemcpy(hostNodes.data(),nodes,capacity*sizeof(Quadtree_node),cudaMemcpyDeviceToHost));
 bool ok=check_quadtree(hostNodes.data(),0,n,&hostPoints,params);std::vector<float> output(2*n);std::vector<std::array<float,2>> sortedInput,sortedOutput;for(int i=0;i<n;i++){output[2*i]=outputX[i];output[2*i+1]=outputY[i];sortedInput.push_back({input[2*i],input[2*i+1]});sortedOutput.push_back({output[2*i],output[2*i+1]});}std::sort(sortedInput.begin(),sortedInput.end());std::sort(sortedOutput.begin(),sortedOutput.end());ok=ok&&sortedInput==sortedOutput;
 save("reports/quadtree-output.bin",output.data(),output.size()*4);
 std::ofstream json("reports/quadtree-nodes.json");json<<"[";int visited=0,leaves=0,leafPoints=0,deepest=0;std::vector<int> membership(n);
 std::function<void(int,int,int,int)> walk=[&](int base,int idx,int levelSize,int depth){const auto& node=hostNodes[base+idx];if(visited++)json<<",";const auto& box=node.bounding_box();const auto lo=box.get_min(),hi=box.get_max();json<<"{\"index\":"<<base+idx<<",\"depth\":"<<depth<<",\"id\":"<<node.id()<<",\"begin\":"<<node.points_begin()<<",\"end\":"<<node.points_end()<<",\"bounds\":["<<lo.x<<","<<lo.y<<","<<hi.x<<","<<hi.y<<"]}";deepest=std::max(deepest,depth);
 if(depth>=maxDepth||node.num_points()<=minPoints){leaves++;leafPoints+=node.num_points();for(int i=node.points_begin();i<node.points_end();i++)membership[i]++;}else for(int c=0;c<4;c++)walk(base+levelSize,4*idx+c,levelSize*4,depth+1);};walk(0,0,1,0);json<<"]\n";for(int count:membership)ok=ok&&count==1;
 checkCudaErrors(cudaFree(points));checkCudaErrors(cudaFree(nodes));printf("{\"passed\":%s,\"points\":%d,\"nodes\":%d,\"leaves\":%d,\"leafPoints\":%d,\"deepestLevel\":%d,\"pointPermutationExact\":%s,\"originalKernel\":true}\n",ok?"true":"false",n,visited,leaves,leafPoints,deepest,sortedInput==sortedOutput?"true":"false");return ok?0:1;
}
