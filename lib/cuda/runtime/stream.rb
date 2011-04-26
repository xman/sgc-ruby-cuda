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


module SGC
module Cuda

class CudaStream

    # Create and return a CUDA stream.
    # @return [CudaStream] A CUDA stream.
    def self.create
        p = FFI::MemoryPointer.new(:CudaStream)
        status = API::cudaStreamCreate(p)
        Pvt::handle_error(status, "Failed to create CUDA stream.")
        new(p)
    end


    # Destroy this CUDA stream.
    def destroy
        status = API::cudaStreamDestroy(self.to_api)
        Pvt::handle_error(status, "Failed to destroy this CUDA stream.")
        API::write_cudastream(@pstream, 0)
        nil
    end


    # @return [Boolean] Return true if all operations in this CUDA stream have completed. Otherwise, return false.
    def query
        status = API::cudaStreamQuery(self.to_api)
        if status == Pvt::CUDA_SUCCESS
            return true
        elsif status == Pvt::CUDA_ERROR_NOT_READY
            return false
        end
        Pvt::handle_error(status, "Failed to query stream.")
        raise CudaStandardError, "Error handling fails to catch this error."
    end


    # Block the calling CPU thread until all operations in this CUDA stream complete.
    # @return [CudaStream] This CUDA stream.
    def synchronize
        status = API::cudaStreamSynchronize(self.to_api)
        Pvt::handle_error(status)
        self
    end


    # Let all future operations submitted to this CUDA stream wait until _event_ complete before beginning execution.
    # @overload wait_event(event)
    # @overload wait_event(event, flags)
    # @param [CudaEvent] event The event to wait for.
    # @param [Integer] flags Currently _flags_ must be set to zero.
    # @return [CudaStream] This CUDA stream.
    def wait_event(event, flags = 0)
        status = API::cudaStreamWaitEvent(self.to_api, event.to_api, flags)
        Pvt::handle_error(status, "Failed to make this CUDA stream's future operations to wait event: flags = #{flags}.")
        self
    end


    # Let all future operations submitted to any CUDA stream wait until _event_ complete before beginning execution.
    # @overload wait_event(event)                      
    # @overload wait_event(event, flags)               
    # @param (see CudaStream#wait_event) 
    def self.wait_event(event, flags = 0)
        status = API::cudaStreamWaitEvent(nil, event.to_api, flags)
        Pvt::handle_error(status, "Failed to make any CUDA stream's future operations to wait event: flags = #{flags}.")
        nil
    end


    # @private
    def initialize(ptr)
        @pstream = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_cudastream(@pstream)
    end

end

# @private
module Pvt

    def self.parse_stream(stream)
        if stream.kind_of?(CudaStream)
            return stream.to_api
        end
        nil
    end

end

end # module
end # module
