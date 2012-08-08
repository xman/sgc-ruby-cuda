require 'rubycu'

include SGC::CU

def ncores_per_sm(major, minor)
    table = {
        0x10 => 8,
        0x11 => 8,
        0x12 => 8,
        0x13 => 8,
        0x20 => 32,
        0x21 => 48,
        0x30 => 192,
    }
    n = table[(major << 4) + minor]
    n or "?"
end


CUInit.init

puts "Number of CUDA devices: #{CUDevice.count}"
puts

(0...CUDevice.count).each do |i|
    dev = CUDevice.get(i)
    cap = dev.compute_capability
    bdim = [dev.attribute(:MAX_BLOCK_DIM_X), dev.attribute(:MAX_BLOCK_DIM_Y), dev.attribute(:MAX_BLOCK_DIM_Z)]
    gdim = [dev.attribute(:MAX_GRID_DIM_X), dev.attribute(:MAX_GRID_DIM_Y), dev.attribute(:MAX_GRID_DIM_Z)]

    puts "CUDA device #{i}"
    puts "CUDA device name                    : #{dev.name}"
    puts "CUDA driver version                 : #{driver_version/1000}.#{driver_version%100}"
    puts "CUDA compute capability             : #{cap[:major]}.#{cap[:minor]}"
    puts "Total global memory                 : #{dev.total_mem/1048576} MB"
    puts "CUDA cores                          : #{mpc = dev.attribute(:MULTIPROCESSOR_COUNT)} x #{nps = ncores_per_sm(cap[:major], cap[:minor])} => #{mpc*nps.to_i}"
    puts "GPU clock rate                      : #{dev.attribute(:CLOCK_RATE)/1000} MHz"
    puts "Memory clock rate                   : #{dev.attribute(:MEMORY_CLOCK_RATE)/1000} MHz"
    puts "Memory bus width                    : #{dev.attribute(:GLOBAL_MEMORY_BUS_WIDTH)}-bit"
    puts "L2 cache size                       : #{dev.attribute(:L2_CACHE_SIZE)} bytes"
    puts "Total constant memory               : #{dev.attribute(:TOTAL_CONSTANT_MEMORY)} bytes"
    puts "Total shared memory per block       : #{dev.attribute(:MAX_SHARED_MEMORY_PER_BLOCK)} bytes"
    puts "Total registers available per block : #{dev.attribute(:MAX_REGISTERS_PER_BLOCK)}"
    puts "Warp size                           : #{dev.attribute(:WARP_SIZE)}"
    puts "Max number of threads per block     : #{dev.attribute(:MAX_THREADS_PER_BLOCK)}"
    puts "Max dimension sizes of a block      : #{bdim[0]} x #{bdim[1]} x #{bdim[2]}"
    puts "Max dimension sizes of a grid       : #{gdim[0]} x #{gdim[1]} x #{gdim[2]}"
    puts "Number of concurrent copy engines   : #{dev.attribute(:ASYNC_ENGINE_COUNT)}"
    puts "Support unified addressing?         : #{dev.attribute(:UNIFIED_ADDRESSING) > 0 ? "Yes" : "No"}"
    puts "Compute mode                        : #{CUComputeMode[dev.attribute(:COMPUTE_MODE)]}"
    puts
end
