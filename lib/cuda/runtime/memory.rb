#
# Copyright (c) 2010-2011 Chung Shin Yee
#
#       shinyee@speedgocomputing.com
#       http://www.speedgocomputing.com
#       http://github.com/xman/sgc-ruby-cuda
#       http://rubyforge.org/projects/rubycuda
#
# This file is part of SGC-Ruby-CUDA.
#
# SGC-Ruby-CUDA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SGC-Ruby-CUDA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SGC-Ruby-CUDA.  If not, see <http://www.gnu.org/licenses/>.
#

require 'cuda/runtime/ffi-cuda'
require 'cuda/runtime/error'
require 'memory/pointer'


module SGC
module Cuda

class CudaDeviceMemory

    # Allocate memory on the device.
    # @param [Integer] nbytes The number of bytes of memory to allocate.
    # @return [*SGC::Memory::MemoryPointer] A memory pointer to the allocated device memory. 
    #
    # @note The returned memory pointer is enabled to call _free_ method on itself.
    def self.malloc(nbytes)
        p = SGC::Memory::MemoryPointer.new
        status = API::cudaMalloc(p.ref, nbytes)
        Pvt::handle_error(status, "Failed to allocate memory on the device: nbytes = #{nbytes}")
        p.instance_eval %{
            def free
                CudaDeviceMemory.free(self)
            end
        }
        p
    end


    # Free the device memory at _devptr_.
    # @param [*SGC::Memory::MemoryPointer] devptr The memory pointer pointing to the device memory to be freed.
    def self.free(devptr)
        status = API::cudaFree(devptr.ptr)
        Pvt::handle_error(status, "Failed to free the device memory.")
        devptr.ptr = 0
        nil
    end

end


module CudaMemory

    # Copy _nbytes_ from the memory at _src_ptr_ to the memory at _dst_ptr_.
    # @param [#ptr] dst_ptr Destination of the memory copy.
    # @param [#ptr] src_ptr Source of the memory copy.
    # @param [Integer] nbytes The number of bytes to copy.
    # @param [Symbol] memcpy_kind The direction of the memory copy specified with one of the following:
    #     * :HOST_TO_HOST
    #     * :HOST_TO_DEVICE
    #     * :DEVICE_TO_HOST
    #     * :DEVICE_TO_DEVICE
    def memcpy(dst_ptr, src_ptr, nbytes, memcpy_kind)
        status = API::cudaMemcpy(dst_ptr.ptr, src_ptr.ptr, nbytes, memcpy_kind)
        Pvt::handle_error(status, "Failed to copy memory.")
    end
    module_function :memcpy


    # Copy _nbytes_ from the host memory at _src_ptr_ to the host memory at _dst_ptr_.
    def memcpy_htoh(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :HOST_TO_HOST)
    end
    module_function :memcpy_htoh

    # Copy _nbytes_ from the host memory at _src_ptr_ to the device memory at _dst_ptr_.
    def memcpy_htod(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :HOST_TO_DEVICE)
    end
    module_function :memcpy_htod

    # Copy _nbytes_ from the device memory at _src_ptr_ to the host memory at _dst_ptr_.
    def memcpy_dtoh(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :DEVICE_TO_HOST)
    end
    module_function :memcpy_dtoh

    # Copy _nbytes_ from the device memory at _src_ptr_ to the device memory at _dst_ptr_.
    def memcpy_dtod(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :DEVICE_TO_DEVICE)
    end
    module_function :memcpy_dtod

end

end # module
end # module
