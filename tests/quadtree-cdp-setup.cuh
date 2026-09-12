// Browser launch setup corresponding to the original sample's host allocations.
// The imported NVIDIA device functions remain unchanged.
__global__ void bind_quadtree_points(Points *points,float *x0,float *y0,float *x1,float *y1){
 Points a(x0,y0);Points b(x1,y1);points[0]=a;points[1]=b;
}
