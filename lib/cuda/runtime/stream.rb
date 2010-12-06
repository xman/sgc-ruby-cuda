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

class CudaStream

    def initialize
        @p = FFI::MemoryPointer.new(:pointer)
    end


    def create
        status = API::cudaStreamCreate(@p)
        Pvt::handle_error(status)
        self
    end


    def destroy
        status = API::cudaStreamDestroy(@p.read_pointer)
        Pvt::handle_error(status)
        @p.write_pointer(0)
        nil
    end


    def query
        status = API::cudaStreamQuery(@p.read_pointer)
        if status == Pvt::CUDA_SUCCESS
            return true
        elsif status == Pvt::CUDA_ERROR_NOT_READY
            return false
        end
        Pvt::hanld_error(status)
        self
    end


    def synchronize
        status = API::cudaStreamSynchronize(@p.read_pointer)
        Pvt::handle_error(status)
        self
    end


    def wait_event(event, flags = 0)
        status = API::cudaStreamWaitEvent(@p.read_pointer, event, flags)
        Pvt::handle_error(status)
        self
    end


    def self.wait_event(event, flags = 0)
        p = FFI::MemoryPointer.new(:pointer)
        p.write_pointer(0)
        status = API::cudaStreamWaitEvent(p.read_pointer, event, flags)
        Pvt::handle_error(status)
        self
    end

    def to_ptr
        @p.read_pointer
    end

end

end # module
end # module
