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
require 'cuda/driver/cu'
require 'cuda/driver/error'
require 'cuda/driver/stream'
require 'helpers/flags'


module SGC
module CU

class CUEvent

    # Create and return an event with _flags_ (CUEventFlags).
    # @overload create
    # @overload create(flags)
    # @return [CUEvent] An event created with _flags_.
    #
    # @example Create events with flags.
    #     CUEvent.create                    #=> event
    #     CUEvent.create(:DEFAULT)          #=> event
    #     CUEvent.create(:BLOCKING_SYNC)    #=> event
    def self.create(*flags)
        flags.empty? == false or flags = :DEFAULT
        p = FFI::MemoryPointer.new(:CUEvent)
        f = CUEventFlags.value(flags)
        status = API::cuEventCreate(p, f)
        Pvt::handle_error(status, "Failed to create event: flags = #{flags}.")
        new(p)
    end


    # Destroy this event.
    def destroy
        status = API::cuEventDestroy(self.to_api)
        Pvt::handle_error(status, "Failed to destroy event.")
        nil
    end


    # @return [Boolean] Return true if this event has been recorded. Otherwise, return false.
    def query
        status = API::cuEventQuery(self.to_api)
        if status == Pvt::CUDA_SUCCESS
            return true
        elsif status == Pvt::CUDA_ERROR_NOT_READY
            return false
        end
        Pvt::handle_error(status, "Failed to query event.")
        raise CUStandardError, "Error handling fails to catch this error."
    end


    # Record this event asynchronously on _stream_.
    # @param [Integer, CUStream] stream The CUDA stream to record this event on.
    #     Setting _stream_ to anything other than an instance of CUStream will record on any stream.
    # @return [CUEvent] This event.
    def record(stream = 0)
        s = Pvt::parse_stream(stream)
        status = API::cuEventRecord(self.to_api, s)
        Pvt::handle_error(status, "Failed to record event.")
        self
    end


    # Block the calling CPU thread until this event has been recorded.
    # @return [CUEvent] This event.
    def synchronize
        status = API::cuEventSynchronize(self.to_api)
        Pvt::handle_error(status, "Failed to synchronize event.")
        self
    end


    # Compute the elapsed time (ms) from _event_start_ (CUEvent) to _event_end_ (CUEvent).
    # @param [CUEvent] event_start The event corresponds to the start time.
    # @param [CUEvent] event_end The event corresponds to the end time.
    # @return [Numeric] The elapsed time in ms.
    def self.elapsed_time(event_start, event_end)
        t = FFI::MemoryPointer.new(:float)
        API::cuEventElapsedTime(t, event_start.to_api, event_end.to_api)
        t.read_float
    end


    # @private
    def initialize(ptr)
        @pevent = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_cuevent(@pevent)
    end

end

end # module
end # module
