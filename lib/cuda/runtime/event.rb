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

class CudaEvent

    # Create and return an event with _flags_.
    # @overload create
    # @overload create(flags)
    # @return [CudaEvent] An event created with _flags_.
    def self.create(*flags)
        flags.empty? == false or flags = :DEFAULT
        p = FFI::MemoryPointer.new(:CudaEvent)
        f = CudaEventFlags.value(flags)
        status = API::cudaEventCreateWithFlags(p, f)
        Pvt::handle_error(status, "Failed to create event: flags = #{flags}")
        new(p)
    end


    # Destroy this event.
    def destroy
        status = API::cudaEventDestroy(self.to_api)
        Pvt::handle_error(status, "Failed to destroy event.")
        API::write_cudaevent(@pevent, 0)
        nil
    end


    # @return [Boolean] Return true if this event has been recorded. Otherwise, return false.
    def query
        status = API::cudaEventQuery(self.to_api)
        if status == Pvt::CUDA_SUCCESS
            return true
        elsif status == Pvt::CUDA_ERROR_NOT_READY
            return false
        end
        Pvt::handle_error(status, "Failed to query event.")
        raise CudaStandardError, "Error handling fails to catch this error."
    end


    # Record this event asynchronously on _stream_.
    # @param [Integer, CudaStream] stream The CUDA stream to record this event on.
    #     Setting _stream_ on anything other than an instance of CudaStream will record on any stream.
    # @return [CudaEvent] This event.
    def record(stream = 0)
        s = Pvt::parse_stream(stream)
        status = API::cudaEventRecord(self.to_api, s)
        Pvt::handle_error(status, "Failed to record event.")
        self
    end


    # Block the calling CPU thread until this event has been recorded.
    # @return [CudaEvent] This event.
    def synchronize
        status = API::cudaEventSynchronize(self.to_api)
        Pvt::handle_error(status)
        self
    end


    # Compute the elapsed time (ms) from _event_start_ to _event_end_.
    # @param [CudaEvent] event_start The event corresponds to the start time.
    # @param [CudaEvent] event_end The event corresponds to the end time.
    # @return [Numeric] The elapsed time in ms.
    def self.elapsed_time(event_start, event_end)
        t = FFI::MemoryPointer.new(:float)
        API::cudaEventElapsedTime(t, event_start.to_api, event_end.to_api)
        t.read_float
    end


    # @private
    def initialize(ptr)
        @pevent = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_cudaevent(@pevent)
    end

end

end # module
end # module
