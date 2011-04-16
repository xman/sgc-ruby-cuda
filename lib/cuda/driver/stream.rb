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


module SGC
module CU

class CUStream

    # Create and return a stream with _flags_.
    # @overload create
    # @overload create(flags)
    # @param [Integer] flags Currently _flags_ must be set to zero.
    # @return [CUStream] A CUDA stream created with _flags_.
    def self.create(flags = 0)
        p = FFI::MemoryPointer.new(:CUStream)
        status = API::cuStreamCreate(p, flags)
        Pvt::handle_error(status, "Failed to create stream: flags = #{flags}.")
        new(p)
    end


    # Destroy this CUDA stream.
    def destroy
        status = API::cuStreamDestroy(self.to_api)
        Pvt::handle_error(status, "Failed to destroy stream.")
        nil
    end


    # @return [Boolean] Return true if all operations in this CUDA stream have completed. Otherwise, return false.
    def query
        status = API::cuStreamQuery(self.to_api)
        if status == Pvt::CUDA_SUCCESS
            return true
        elsif status == Pvt::CUDA_ERROR_NOT_READY
            return false
        end
        Pvt::hanld_error(status, "Failed to query stream.")
        raise CUStandardError, "Error handling fails to catch this error."
    end


    # Block the calling CPU thread until all operations in this CUDA stream complete.
    # @return [CUStream] This CUDA stream.
    def synchronize
        status = API::cuStreamSynchronize(self.to_api)
        Pvt::handle_error(status, "Failed to synchronize stream.")
        self
    end


    # Let all future operations submitted to this CUDA stream wait until _event_ (CUEvent) complete before beginning execution.
    # @overload wait_event(event)
    # @overload wait_event(event, flags)
    # @param [CUEvent] event The event to wait for.
    # @param [Integer] flags Currently _flags_ must be set to zero.
    # @return [CUStream] This CUDA stream.
    def wait_event(event, flags = 0)
        status = API::cuStreamWaitEvent(self.to_api, event.to_api, flags)
        Pvt::handle_error(status, "Failed to make stream's future operations to wait event: flags = #{flags}.")
        self
    end


    # Let all future operations submitted to stream 0 (NULL stream) wait until _event_ (CUEvent) complete before beginning execution.
    # @overload wait_event(event)
    # @overload wait_event(event, flags)
    # @param (see CUStream#wait_event)
    def self.wait_event(event, flags = 0)
        status = API::cuStreamWaitEvent(nil, event.to_api, flags)
        Pvt::handle_error(status, "Failed to make current stream's future operations to wait event: flags = #{flags}.")
        nil
    end


    # @private
    def initialize(ptr)
        @pstream = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_custream(@pstream)
    end

end

# @private
module Pvt

    def self.parse_stream(stream)
        if stream.kind_of?(CUStream)
            return stream.to_api
        end
        nil
    end

end

end # module
end # module
