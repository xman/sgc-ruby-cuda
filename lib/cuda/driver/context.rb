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
require 'helpers/flags'


module SGC
module CU

class CUContext

    # Create a new CUDA context with _flags_ (CUContextFlags) and _device_ (CUDevice),
    # then associate it with the calling thread, and return the context.
    #
    # @overload create(device)
    # @overload create(flags, device)
    # @param [Integer, CUContextFlags, Array<Integer, CUContextFlags>] flags
    #   The list of flags to use for the CUDA context creation.
    #   Setting _flags_ to 0 or ommitting _flags_ uses SCHED_AUTO.
    # @param [CUDevice] device The device to create the CUDA context with.
    # @return [CUContext] A CUDA context created with _flags_ and _device_.
    #
    # @example Create CUDA context with different flags.
    #     dev = CUDevice.get(0)
    #     CUContext.create(dev)                 #=> ctx
    #     CUContext.create(0, dev)              #=> ctx
    #     CUContext.create(:SCHED_SPIN, dev)    #=> ctx
    #     CUContext.create([:SCHED_SPIN, :BLOCKING_SYNC], dev)    #=> ctx
    def self.create(arg1, arg2 = nil)
        if arg2 != nil
            flags, dev = arg1, arg2
            flags = CUContextFlags.value(flags)
        else
            flags = 0
            dev = arg1
        end

        p = FFI::MemoryPointer.new(:CUContext)
        status = API::cuCtxCreate(p, flags, dev.to_api)
        Pvt::handle_error(status, "Failed to create CUDA context: flags = #{flags}.")
        new(p)
    end


    # Destroy this CUDA context.
    def destroy
        status = API::cuCtxDestroy(self.to_api)
        Pvt::handle_error(status, "Failed to destroy CUDA context.")
        nil
    end


    # @deprecated
    #
    # Increment the reference count on this CUDA context.
    # @overload attach
    # @overload attach(flags)
    # @param [Integer] flags Currently _flags_ must be set to zero.
    # @return [CUContext] This CUDA context.
    def attach(flags = 0)
        status = API::cuCtxAttach(@pcontext, flags)
        Pvt::handle_error(status, "Failed to attach CUDA context: flags = #{flags}.")
        self
    end


    # @deprecated
    #
    # Decrement the reference count on this CUDA context.
    def detach
        status = API::cuCtxDetach(self.to_api)
        Pvt::handle_error(status, "Failed to detach CUDA context.")
        nil
    end


    # @return [CUContext] The CUDA context bound to the calling CPU thread.
    def self.current
        p = FFI::MemoryPointer.new(:CUContext)
        status = API::cuCtxGetCurrent(p)
        Pvt::handle_error(status, "Failed to get the current CUDA context.")
        new(p)
    end


    # Set the current CUDA context to _context_.
    # @param [CUContext] The CUDA context to set as the current CUDA context.
    def self.current=(context)
        status = API::cuCtxSetCurrent(context.to_api)
        Pvt::handle_error(status, "Failed to set the current CUDA context.")
    end


    # Push this CUDA context onto the CUDA context stack, which becomes currently active CUDA context.
    # @return [CUContext] This CUDA context.
    def push_current
        status = API::cuCtxPushCurrent(self.to_api)
        Pvt::handle_error(status, "Failed to push this CUDA context.")
        self
    end


    # @return [Integer] The API version used to create this CUDA context.
    def api_version
        p = FFI::MemoryPointer.new(:uint)
        status = API::cuCtxGetApiVersion(self.to_api, p)
        Pvt::handle_error(status, "Failed to get the API version of this CUDA context.")
        p.get_uint(0)
    end


    # @return [Integer] The API version used to create the current CUDA context.
    def self.api_version
        p = FFI::MemoryPointer.new(:uint)
        status = API::cuCtxGetApiVersion(nil, p)
        Pvt::handle_error(status, "Failed to get the API version of the current CUDA context.")
        p.get_uint(0)
    end


    # @return [CUDevice] The device associated to the current CUDA context.
    def self.device
        p = FFI::MemoryPointer.new(:CUDevice)
        status = API::cuCtxGetDevice(p)
        Pvt::handle_error(status, "Failed to get the current CUDA context's device.")
        CUDevice.send(:new, p)
    end


    # @param [CULimit] lim The particular limit attribute to query.
    # @return [Integer] The limit _lim_ (CULimit) of the current CUDA context.
    #
    # @example Get the stack size limit.
    #     CUContext.limit(:STACK_SIZE)    #=> 8192
    def self.limit(lim)
        p = FFI::MemoryPointer.new(:size_t)
        status = API::cuCtxGetLimit(p, lim)
        Pvt::handle_error(status, "Failed to get the current CUDA context limit: limit = #{lim}")
        API::read_size_t(p)
    end


    # Set the limit _lim_ (CULimit) of the current CUDA context to _value_.
    # @param [CULimit] lim The particular limit attribute to set.
    # @param [Integer] value The value to set the limit to.
    #
    # @example Set the stack size limit.
    #     CUContext.limit = [:STACK_SIZE, 8192]    #=> [:STACK_SIZE, 8192]
    #     CUContext.limit = :STACK_SIZE, 8192      #=> [:STACK_SIZE, 8192]
    def self.limit=(*lim_val_pair)
        lim, val = lim_val_pair.flatten
        lim != nil && val != nil or raise ArgumentError, "Invalid limit and value pair given: limit = #{lim}, value = #{val}."
        status = API::cuCtxSetLimit(lim, val)
        Pvt::handle_error(status, "Failed to set the current CUDA context limit: limit = #{lim}, value = #{val}")
    end


    # @return [CUFunctionCache] The cache config of the current CUDA context.
    #
    # @example Get the cache config.
    #     CUContext.cache_config    #=> :PREFER_NONE
    def self.cache_config
        p = FFI::MemoryPointer.new(:enum)
        status = API::cuCtxGetCacheConfig(p)
        Pvt::handle_error(status, "Failed to get the current CUDA context cache config.")
        CUFunctionCache[API::read_enum(p)]
    end


    # Set the cache to _conf_ (CUFunctionCache) for the current CUDA context.
    #
    # @example Set the cache config to prefer shared.
    #     CUContext.cache_config = :PREFER_SHARED    #=> :PREFER_SHARED
    def self.cache_config=(conf)
        status = API::cuCtxSetCacheConfig(conf)
        Pvt::handle_error(status, "Failed to set the current CUDA context cache config: config = #{conf}")
    end


    # Pop the current CUDA context from the CUDA context stack, which becomes inactive.
    def self.pop_current
        p = FFI::MemoryPointer.new(:CUContext)
        status = API::cuCtxPopCurrent(p)
        Pvt::handle_error(status, "Failed to pop current context.")
        new(p)
    end


    # Block until all the tasks of the current CUDA context complete.
    def self.synchronize
        status = API::cuCtxSynchronize
        Pvt::handle_error(status, "Failed to synchronize the current context.")
        nil
    end


    # Enable the current context to access the memory of the peer context.
    # @param [CUContext] peer_context The peer context's memory to be accessed.
    # @param [Integer] flags Currently flags must be set to zero.
    # @return [Class] This class.
    #
    # @since CUDA 4.0
    def self.enable_peer_access(peer_context, flags = 0)
        status = API::cuCtxEnablePeerAccess(peer_context.to_api, flags)
        Pvt::handle_error(status, "Failed to enable peer access: flags = #{flags}.")
        self
    end


    # Disable the current context from accessing the memory of the peer context.
    # @param [CUContext] peer_context The peer context.
    # @return [Class] This class.
    #
    # @since CUDA 4.0
    def self.disable_peer_access(peer_context)
        status = API::cuCtxDisablePeerAccess(peer_context.to_api)
        Pvt::handle_error(status, "Failed to disable peer access.")
        self
    end


    # @private
    def initialize(ptr)
        @pcontext = ptr
    end
    private_class_method(:new)


    # @private
    def to_api
        API::read_cucontext(@pcontext)
    end

end

end # module
end # module
