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
require 'cuda/driver/context'


module SGC
module CU

class CUDevice

    # @return [Integer] The number of CUDA devices.
    def self.count
        p = FFI::MemoryPointer.new(:int)
        status = API::cuDeviceGetCount(p)
        Pvt::handle_error(status, "Failed to get device count.")
        p.read_int
    end


    # @param [Integer] index The index (0..CUDevice.count-1) of the device to get.
    # @return [CUDevice] The device corresponding to CUDA device _index_.
    def self.get(index)
        p = FFI::MemoryPointer.new(:CUDevice)
        status = API::cuDeviceGet(p, index)
        Pvt::handle_error(status, "Failed to get device #{index}.")
        new(p)
    end


    # @return [String] The name of this device with a maximum of 255 characters.
    def name
        s = FFI::MemoryPointer.new(:char, 256)
        status = API::cuDeviceGetName(s, 256, self.to_api)
        Pvt::handle_error(status, "Failed to get device name.")
        s.read_string
    end


    # @return [Hash{ :major, :minor }] The compute capability of this device.
    #
    # @example For a device with compute capability 1.3:
    #     dev.compute_capability    #=> { major: 1, minor: 3 }
    def compute_capability
        cap = FFI::MemoryPointer.new(:int, 2)
        status = API::cuDeviceComputeCapability(cap[0], cap[1], self.to_api)
        Pvt::handle_error(status, "Failed to query device compute capability.")
        { major: cap[0].read_int, minor: cap[1].read_int }
    end


    # @param [CUDeviceAttribute] attrib The particular attribute of this device to query.
    # @return [Integer] The attribute _attrib_ of this device.
    #
    # @example
    #     dev.attribute(:MAX_THREADS_PER_BLOCK)        #=> 512
    #     dev.attribute(:MULTIPROCESSOR_COUNT)         #=> 30
    #     dev.attribute(:MAX_SHARED_MEMORY_PER_BLOCK)  #=> 16384
    def attribute(attrib)
        p = FFI::MemoryPointer.new(:int)
        status = API::cuDeviceGetAttribute(p, attrib, self.to_api)
        Pvt::handle_error(status, "Failed to query device attribute #{attrib}.")
        p.read_int
    end


    # @return [Hash] The properties of this device in a hash with the following keys:
    #     * :clock_rate
    #     * :max_grid_size
    #     * :max_threads_dim
    #     * :max_threads_per_block
    #     * :mem_pitch
    #     * :regs_per_block
    #     * :shared_mem_per_block
    #     * :simd_width
    #     * :texture_align
    #     * :total_constant_memory
    def properties
        prop = API::CUDevProp.new
        status = API::cuDeviceGetProperties(prop.to_ptr, self.to_api)
        Pvt::handle_error(status, "Failed to get device properties.")
        h = {}
        h[:clock_rate] = prop[:clockRate]
        h[:max_grid_size] = prop[:maxGridSize]
        h[:max_threads_dim] = prop[:maxThreadsDim]
        h[:max_threads_per_block] = prop[:maxThreadsPerBlock]
        h[:mem_pitch] = prop[:memPitch]
        h[:regs_per_block] = prop[:regsPerBlock]
        h[:shared_mem_per_block] = prop[:sharedMemPerBlock]
        h[:simd_width] = prop[:SIMDWidth]
        h[:texture_align] = prop[:textureAlign]
        h[:total_constant_memory] = prop[:totalConstantMemory]
        h
    end


    # @return [Integer] The total amount of device memory in bytes.
    def total_mem
        p = FFI::MemoryPointer.new(:size_t)
        status = API::cuDeviceTotalMem(p, self.to_api)
        Pvt::handle_error(status, "Failed to get device total amount of memory available.")
        API::read_size_t(p)
    end


    # Allocate _nbytes_ of device memory from the current device.
    # @param [Integer] nbytes The number of bytes to allocate.
    # @return [CUDevicePtr] A device pointer to the allocated memory.
    def self.malloc(nbytes)
        p = FFI::MemoryPointer.new(:CUDevicePtr)
        status = API::cuMemAlloc(p, nbytes)
        Pvt::handle_error(status, "Failed to allocate device memory: size = #{nbytes}.")
        CUDevicePtr.send(:new, p)
    end


    # @param [CUDevice] dev The device which is to access the memory of the device _peer_dev_.
    # @param [CUDevice] peer_dev The device which its memory is to be accessed by the device _dev_.
    # @return [Boolean] True if device _dev_ may directly access the memory of device _peer_dev_.
    #
    # @since CUDA 4.0
    def self.can_access_peer?(dev, peer_dev = nil)
        # TODO: Remove the following workaround for JRuby when the default argument bug is fixed.
        if peer_dev.nil?
            peer_dev = dev
            dev = CUContext.device
        end
        b = FFI::MemoryPointer.new(:int)
        status = API::cuDeviceCanAccessPeer(b, dev.to_api, peer_dev.to_api)
        Pvt::handle_error(status, "Failed to query can access peer.")
        b.read_int == 1 ? true : false
    end


    # @private
    def initialize(ptr)
        @pdev = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_cudevice(@pdev)
    end

end

end # module
end # module
