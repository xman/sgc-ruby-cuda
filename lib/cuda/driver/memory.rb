#
# Copyright (c) 2011 Chung Shin Yee
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

require 'cuda/driver/ffi-cu'
require 'cuda/driver/error'
require 'cuda/driver/stream'


module SGC
module CU

module CUMemory

    # Copy _nbytes_ from the memory at _src_ptr_ to the memory at _dst_ptr_.
    # The type of memory (host or device) is inferred from the pointer value.
    def memcpy(dst_ptr, src_ptr, nbytes)
        status = API::cuMemcpy(dst_ptr.to_api, src_ptr.to_api, nbytes)
        Pvt::handle_error(status, "Failed to copy memory: size = #{nbytes}")
        nil
    end
    module_function :memcpy


    # Copy _nbytes_ from the memory at _src_ptr_ to the memory at _dst_ptr_ on _stream_ asynchronously.
    # The type of memory (host or device) is inferred from the pointer value.
    def memcpy_async(dst_ptr, src_ptr, nbytes, stream)
        s = Pvt::parse_stream(stream)
        status = API::cuMemcpyAsync(dst_ptr.to_api, src_ptr.to_api, nbytes, s)
        Pvt::handle_error(status, "Failed to copy memory asynchronously: size = #{nbytes}")
        nil
    end
    module_function :memcpy_async


    # Copy _nbytes_ from the host memory at _src_mem_ to the device memory at _dst_devptr_.
    def memcpy_htod(dst_devptr, src_mem, nbytes)
        status = API::cuMemcpyHtoD(dst_devptr.to_api, src_mem.ptr, nbytes)
        Pvt::handle_error(status, "Failed to copy memory from host to device: size = #{nbytes}")
        nil
    end
    module_function :memcpy_htod


    # Copy _nbytes_ from the host memory at _src_mem_ to the device memory at _dst_devptr_ on _stream_ asynchronously.
    #
    # @note The _src_mem_ should be *page-locked* memory.
    # @note Not implemented yet.
    def memcpy_htod_async(dst_devptr, src_mem, nbytes, stream)
        s = Pvt::parse_stream(stream)
        status = API::cuMemcpyHtoDAsync(dst_devptr.to_api, src_mem.ptr, nbytes, s)
        Pvt::handle_error(status, "Failed to copy memory from host to device asynchronously: size = #{nbytes}")
        nil
    end
    module_function :memcpy_htod_async


    # Copy _nbytes_ from the device memory at _src_devptr_ to the host memory at _dst_mem_.
    def memcpy_dtoh(dst_mem, src_devptr, nbytes)
        status = API::cuMemcpyDtoH(dst_mem.ptr, src_devptr.to_api, nbytes)
        Pvt::handle_error(status, "Failed to copy memory from device to host: size = #{nbytes}")
        nil
    end
    module_function :memcpy_dtoh


    # Copy _nbytes_ from the device memory at _src_devptr_ to the host memory at _dst_mem_ on _stream_ asynchronously.
    #
    # @note The _dst_mem_ should be *page-locked* memory.
    # @note Not implemented yet.
    def memcpy_dtoh_async(dst_mem, src_devptr, nbytes, stream)
        s = Pvt::parse_stream(stream)
        status = API::cuMemcpyDtoHAsync(dst_mem.ptr, src_devptr.to_api, nbytes, s)
        Pvt::handle_error(status, "Failed to copy memory from device to host asynchronously: size = #{nbytes}")
        nil
    end
    module_function :memcpy_dtoh_async


    # Copy _nbytes_ from the device memory at _src_devptr_ to the device memory at _dst_devptr_ asynchronously.
    def memcpy_dtod(dst_devptr, src_devptr, nbytes)
        status = API::cuMemcpyDtoD(dst_devptr.to_api, src_devptr.to_api, nbytes)
        Pvt::handle_error(status, "Failed to copy memory from device to device asynchronously: size = #{nbytes}.")
        nil
    end
    module_function :memcpy_dtod


    # Copy _nbytes_ from the device memory at _src_devptr_ to the device memory at _dst_devptr_ on _stream_ asynchronously.
    #
    # @note Not implemented yet.
    def memcpy_dtod_async(dst_devptr, src_devptr, nbytes, stream)
        s = Pvt::parse_stream(stream)
        status = API::cuMemcpyDtoDAsync(dst_devptr.to_api, src_devptr.to_api, nbytes, s)
        Pvt::handle_error(status, "Failed to copy memory from device to device asynchronously: size = #{nbytes}.")
        nil
    end
    module_function :memcpy_dtod_async


    # @return [Hash{ :free, :total }] A hash with the amount of free and total device memory in bytes.
    def mem_info
        pfree = FFI::MemoryPointer.new(:size_t)
        ptotal = FFI::MemoryPointer.new(:size_t)
        status = API::cuMemGetInfo(pfree, ptotal)
        Pvt::handle_error(status, "Failed to get memory information.")
        { free: API::read_size_t(pfree), total: API::read_size_t(ptotal) }
    end
    module_function :mem_info

end

end # module
end # module
