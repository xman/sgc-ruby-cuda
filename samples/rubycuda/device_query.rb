require 'rubycuda'

include SGC::Cuda

def ncores_per_sm(major, minor)
    table = {
        0x10 => 8,
        0x11 => 8,
        0x12 => 8,
        0x13 => 8,
        0x20 => 32,
        0x21 => 48,
    }
    n = table[(major << 4) + minor]
    n or "?"
end


puts "Number of CUDA devices: #{CudaDevice.count}"
puts

(0...CudaDevice.count).each do |i|
    CudaDevice.current = i
    prop = CudaDevice.properties

    puts "CUDA device #{i}"
    puts "CUDA device name                    : #{prop.name}"
    puts "CUDA driver version                 : #{driver_version/1000}.#{driver_version%100}"
    puts "CUDA runtime version                : #{runtime_version/1000}.#{runtime_version%100}"
    puts "CUDA compute capability             : #{prop.major}.#{prop.minor}"
    puts "Total global memory                 : #{prop.total_global_mem/1048576} MB"
    puts "CUDA cores                          : #{mpc = prop.multi_processor_count} x #{nps = ncores_per_sm(prop.major, prop.minor)} => #{mpc*nps.to_i}"
    puts "GPU clock rate                      : #{prop.clock_rate/1000} MHz"
    puts "Memory clock rate                   : #{prop.memory_clock_rate/1000} MHz"
    puts "Memory bus width                    : #{prop.memory_bus_width}-bit"
    puts "L2 cache size                       : #{prop.l2_cache_size} bytes"
    puts "Total constant memory               : #{prop.total_const_mem} bytes"
    puts "Total shared memory per block       : #{prop.shared_mem_per_block} bytes"
    puts "Total registers available per block : #{prop.regs_per_block}"
    puts "Warp size                           : #{prop.warp_size}"
    puts "Max number of threads per block     : #{prop.max_threads_per_block}"
    puts "Max dimension sizes of a block      : #{prop.max_threads_dim[0]} x #{prop.max_threads_dim[1]} x #{prop.max_threads_dim[2]}"
    puts "Max dimension sizes of a grid       : #{prop.max_grid_size[0]} x #{prop.max_grid_size[1]} x #{prop.max_grid_size[2]}"
    puts "Number of concurrent copy engines   : #{prop.async_engine_count}"
    puts "Support unified addressing?         : #{prop.unified_addressing > 0 ? "Yes" : "No"}"
    puts "Compute mode                        : #{CudaComputeMode[prop.compute_mode]}"
    puts
end
