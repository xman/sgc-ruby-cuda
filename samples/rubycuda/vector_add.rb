require 'tempfile'
require 'rbconfig'
require 'rubycuda'

include SGC::Cuda
include SGC::Cuda::CudaMemory


# @todo Use internal compiler helpers once they are ready.
def compile(src_str)
    src_file = Tempfile.new(["kernel_src", ".cu"])
    src_file.write(src_str)
    src_file.close

    out_file = Tempfile.new(["kernel", ".so"])
    out_file.close

    case Config::CONFIG['target_os']
        when /darwin/    # Build universal binary for i386 and x86_64 platforms.
            f32 = Tempfile.new(["kernel32", ".so"])
            f64 = Tempfile.new(["kernel64", ".so"])
            f32.close
            f64.close
            system %{nvcc -shared -m32 -Xcompiler -fPIC #{src_file.path} -o #{f32.path}}
            system %{nvcc -shared -m64 -Xcompiler -fPIC #{src_file.path} -o #{f64.path}}
            system %{lipo -arch i386 #{f32.path} -arch x86_64 #{f64.path} -create -output #{out_file.path}}
        else    # Build default platform binary.
            system %{nvcc -shared -Xcompiler -fPIC #{src_file.path} -o #{out_file.path}}
    end

    out_file
end


vadd_kernel_src = <<EOS
extern "C" {
    __global__ void vadd(const float* a, const float* b, float* c, int n)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n)
            c[i] = a[i] + b[i];
    }
}
EOS


CudaDevice.current = ENV['DEVID'].to_i if ENV['DEVID']

puts "Vector Addition"
N = 50000

# Allocate host buffers.
ha = Buffer.new(:float, N)
hb = Buffer.new(:float, N)
hc = Buffer.new(:float, N)

# Allocate device buffers.
nbytes = N*Buffer.element_size(:float)
da = CudaDeviceMemory.malloc(nbytes)
db = CudaDeviceMemory.malloc(nbytes)
dc = CudaDeviceMemory.malloc(nbytes)

# Initialize host buffers.
(0...N).each do |i|
    ha[i] = rand
    hb[i] = rand
    hc[i] = 0
end

# Prepare and load vadd kernel.
kernel_lib_file = compile(vadd_kernel_src)
CudaFunction.load_lib_file(kernel_lib_file.path)

# Copy input buffers from host memory to device memory.
memcpy_htod(da, ha, nbytes)
memcpy_htod(db, hb, nbytes)

# Invoke vadd kernel.
nthreads_per_block = 256
block_dim = Dim3.new(nthreads_per_block)
grid_dim = Dim3.new((N + nthreads_per_block - 1) / nthreads_per_block)
CudaFunction.configure(block_dim, grid_dim)
CudaFunction.setup(da, db, dc, N)
f = CudaFunction.new("vadd")
f.launch

# Copy output buffer from device memory to host memory.
memcpy_dtoh(hc, dc, nbytes)

# Verify result.
all_matches = true
(0...N).each do |i|
    sum = ha[i] + hb[i]
    if (hc[i] - sum).abs > 1e-5
        puts "Result a[#{i}] + b[#{i}] does not match. Expect #{sum}, but we got #{hc[i]}"
        all_matches = false
        break
    end
end
puts "Verification completed. All matches? #{all_matches ? "YES" : "NO"}"

CudaFunction.unload_all_libs

# Free device buffers.
da.free
db.free
dc.free
