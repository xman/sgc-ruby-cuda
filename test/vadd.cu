extern "C" {

texture<int, 1, cudaReadModeElementType> tex;
__device__ int gvar = 1997;

__global__ void vadd(const int* a, const int* b, int* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bb = tex1Dfetch(tex, i);
        c[i] = a[i] + bb;
    }
}

__global__ void vaddf(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

}
