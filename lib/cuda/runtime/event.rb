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


module SGC
module Cuda

class CudaEvent

    def initialize
        @p = FFI::MemoryPointer.new(:pointer)
    end


    def create(flags = CUDA_EVENT_DEFAULT)
        if flags == CUDA_EVENT_DEFAULT
            status = API::cudaEventCreate(@p)
        else
            flags = CudaEventFlags[flags] if flags.is_a?(Symbol)
            status = API::cudaEventCreateWithFlags(@p, flags)
        end
        Pvt::handle_error(status)
        self
    end


    def destroy
        status = API::cudaEventDestroy(@p.read_pointer)
        Pvt::handle_error(status)
        @p.write_pointer(0)
        nil
    end


    def query
        status = API::cudaEventQuery(@p.read_pointer)
        if status == Pvt::CUDA_SUCCESS
            return true
        elsif status == Pvt::CUDA_ERROR_NOT_READ
            return false
        end
        Pvt::handle_error(status)
        self
    end


    def record(stream = 0)
        if stream == 0
            p = FFI::MemoryPointer.new(:pointer)
            p.write_pointer(0)
            stream = p.read_pointer
        else
            stream = stream.to_ptr
        end
        status = API::cudaEventRecord(@p.read_pointer, stream)
        Pvt::handle_error(status)
        self
    end


    def synchronize
        status = API::cudaEventSynchronize(@p.read_pointer)
        Pvt::handle_error(status)
        self
    end


    def to_ptr
        @p.read_pointer
    end


    def self.elapsed_time(event_start, event_end)
        t = FFI::MemoryPointer.new(:float)
        API::cudaEventElapsedTime(t, event_start.to_ptr, event_end.to_ptr)
        t.read_float
    end

protected

    CUDA_EVENT_DEFAULT = CudaEventFlags[:cudaEventDefault]

end

end # module
end # module
