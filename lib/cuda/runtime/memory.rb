#
# Copyright (c) 2010 Chung Shin Yee
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

    def self.malloc(nbytes)
        p = SGC::Memory::MemoryPointer.new
        status = API::cudaMalloc(p.ref, nbytes)
        Pvt::handle_error(status)
        p
    end


    def self.free(devptr)
        status = API::cudaFree(devptr.ptr)
        Pvt::handle_error(status)
        nil
    end

end


module CudaMemory

    def memcpy(dst_ptr, src_ptr, nbytes, memcpy_kind)
        status = API::cudaMemcpy(dst_ptr.ptr, src_ptr.ptr, nbytes, memcpy_kind)
        Pvt::handle_error(status)
    end
    module_function :memcpy

    def memcpy_htoh(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :cudaMemcpyHostToHost)
    end
    module_function :memcpy_htoh

    def memcpy_htod(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :cudaMemcpyHostToDevice)
    end
    module_function :memcpy_htod

    def memcpy_dtoh(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :cudaMemcpyDeviceToHost)
    end
    module_function :memcpy_dtoh

    def memcpy_dtod(dst_ptr, src_ptr, nbytes)
        memcpy(dst_ptr, src_ptr, nbytes, :cudaMemcpyDeviceToDevice)
    end
    module_function :memcpy_dtod

end

end # module
end # module
