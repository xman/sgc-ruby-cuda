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
require 'cuda/driver/context'
require 'memory/pointer'


module SGC
module CU

class CUDevicePtr

    # Free the allocated device memory that this pointer pointing to.
    def free
        status = API::cuMemFree(API::read_cudeviceptr(@pdevptr))
        Pvt::handle_error(status, "Failed to free device memory.")
        API::write_cudeviceptr(@pdevptr, 0)
        nil
    end


    # @param [Integer] index Number of bytes to offset from this pointer address.
    # @return [CUDevicePtr] A pointer pointing to the memory location _index_ (bytes)
    #   from this pointer address.
    def offset(index)
        p = FFI::MemoryPointer.new(:CUDevicePtr)
        addr = API::read_cudeviceptr(@pdevptr).to_i + index
        API::write_cudeviceptr(p, addr)
        CUDevicePtr.send(:new, p)
    end


    def attribute(attrib)
        case attrib
        when :CONTEXT
            p = FFI::MemoryPointer.new(:CUContext)
            status = API::cuPointerGetAttribute(p, attrib, self.to_api)
            Pvt::handle_error(status, "Failed to get pointer context.")
            r = CUContext.send(:new, p)
        when :MEMORY_TYPE
            p = FFI::MemoryPointer.new(:uint)
            status = API::cuPointerGetAttribute(p, attrib, self.to_api)
            Pvt::handle_error(status, "Failed to get pointer memory type.")
            r = CUMemoryType[p.read_uint]
        when :DEVICE_POINTER
            p = FFI::MemoryPointer.new(:CUDevicePtr)
            status = API::cuPointerGetAttribute(p, attrib, self.to_api)
            Pvt::handle_error(status, "Failed to get device pointer.")
            r = CUDevicePtr.send(:new, p)
        when :HOST_POINTER
            p = FFI::MemoryPointer.new(:pointer)
            status = API::cuPointerGetAttribute(p, attrib, self.to_api)
            Pvt::handle_error(status, "Failed to get host pointer.")
            r = SGC::Memory::MemoryPointer.new(p.read_pointer)
        else
            raise TypeError, "Expect _attrib_ one of #{CUPointerAttribute.symbols}, but we get #{attrib}."
        end
        r
    end


    # @private
    def initialize(ptr)
        @pdevptr = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_cudeviceptr(@pdevptr)
    end

end

end # module
end # module
