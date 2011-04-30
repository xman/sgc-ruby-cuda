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

class CudaDevice

    # @return [Integer] The number of CUDA devices.
    def self.count
        p = FFI::MemoryPointer.new(:int)
        status = API::cudaGetDeviceCount(p)
        Pvt::handle_error(status, "Failed to get device count.")
        p.read_int
    end


    # @return [Integer] The index of the current CUDA device in use.
    def self.get
        p = FFI::MemoryPointer.new(:int)
        status = API::cudaGetDevice(p)
        Pvt::handle_error(status, "Failed to get current device.")
        p.read_int
    end
    class << self; alias_method :current, :get; end


    # Set _devid_ as the current CUDA device.
    # @param [Integer] devid The index (0..CudaDevice.count-1) of the CUDA device to set as current.
    # @return [Class] This class.
    def self.set(devid)
        status = API::cudaSetDevice(devid)
        Pvt::handle_error(status, "Failed to set current device: devid = #{devid}.")
        self
    end
    class << self; alias_method :current=, :set; end


    # @param [CudaDeviceProp] prop The criteria for choosing a CUDA device.
    # @return [Integer] The index of the CUDA device best matches the criteria.
    def self.choose(prop)
        pdev = FFI::MemoryPointer.new(:int)
        status = API::cudaChooseDevice(pdev, prop.to_ptr)
        Pvt::handle_error(status, "Failed to choose a device with criteria.")
        pdev.read_int
    end


    # @param [Integer] devid The index of the device to query.
    # @return [CudaDeviceProp] The properties of the device _devid_.
    def self.properties(devid = self.get)
        prop = CudaDeviceProp.new
        status = API::cudaGetDeviceProperties(prop.to_ptr, devid)
        Pvt::handle_error(status, "Failed to get device properties: devid = #{devid}.")
        prop
    end


    # Set the flags to be used for device execution.
    # @param [Integer, CudaDeviceFlags, Array<Integer, CudaDeviceFlags>] flags The flags for device execution.
    def self.flags=(flags)
        f = CudaDeviceFlags.value(flags)
        status = API::cudaSetDeviceFlags(f)
        Pvt::handle_error(status, "Failed to set device flags: flags = #{flags}.")
    end


    # Set the list of CUDA devices that can be used.
    # @param [Array] devs The list of CUDA device indexes.
    def self.valid_devices=(devs)
        p = FFI::MemoryPointer.new(:int, devs.count)
        devs.each_with_index do |devid, i|
            p[i].write_int(devid)
        end
        status = API::cudaSetValidDevices(p, devs.count)
        Pvt::handle_error(status, "Failed to set valid devices: devs = #{devs}.")
    end


    # @return [CudaFunctionCache] The cache config of the current CUDA device.
    #
    # @since CUDA 4.0
    def self.cache_config
        p = FFI::MemoryPointer.new(:enum)
        status = API::cudaDeviceGetCacheConfig(p)
        Pvt::handle_error(status, "Failed to get the current CUDA device cache config.")
        CudaFunctionCache[API::read_enum(p)]
    end


    # Set the cache config of the current CUDA device to _conf_.
    # @param [CudaFunctionCache] conf The cache config of the current CUDA device to set to.
    #
    # @since CUDA 4.0
    def self.cache_config=(conf)
        status = API::cudaDeviceSetCacheConfig(conf)
        Pvt::handle_error(status, "Failed to set the current CUDA device cache config.")
    end


    # @param [CudaLimit] lim The particular limit attribute to query.
    # @return [CudaLimit] The limit _lim_ of the current CUDA device.
    #
    # @since CUDA 4.0
    def self.limit(lim)
        p = FFI::MemoryPointer.new(:size_t)
        status = API::cudaDeviceGetLimit(p, lim)
        Pvt::handle_error(status, "Failed to get the current CUDA device limit: limit = #{lim}.")
        API::read_size_t(p)
    end


    # Set the limit _lim_ of the current CUDA device.
    # @param [CudaLimit] lim The particular limit attribute to set.
    # @param [Integer] value The value to set the limit to.
    #
    # @since CUDA 4.0
    def self.limit=(*lim_val_pair)
        lim, val = lim_val_pair.flatten
        lim != nil && val != nil or raise ArgumentError, "Invalid limit and value pair given: limit = #{lim}, value = #{val}."
        status = API::cudaDeviceSetLimit(lim, val)
        Pvt::handle_error(status, "Failed to set the current CUDA device limit: limit = #{lim}, value = #{val}")
    end


    # Destroy all allocations and reset all state on the current CUDA device.
    # @return [Class] This class.
    #
    # @since CUDA 4.0
    def self.reset
        status = API::cudaDeviceReset()
        Pvt::handle_error(status, "Failed to reset the current CUDA device.")
        self
    end


    # Block until all the tasks of the current CUDA device complete.
    # @return [Class] This class.
    #
    # @since CUDA 4.0
    def self.synchronize
        status = API::cudaDeviceSynchronize()
        Pvt::handle_error(status, "Failed to synchronize the current CUDA device.")
        self
    end


    # @param [Integer] devid The device's ID which is to access the memory of the device _peer_devid_.
    # @param [Integer] peer_devid The device's ID which its memory is to be accessed by the device _devid_.
    # @return [Boolean] True if device _devid_ is capable of directly accessing memory from device _peer_devid_.
    #
    # @since CUDA 4.0
    def self.can_access_peer?(devid = self.get, peer_devid)
        b = FFI::MemoryPointer.new(:int)
        status = API::cudaDeviceCanAccessPeer(b, devid, peer_devid)
        Pvt::handle_error(status, "Failed to query can access peer: devid = #{devid}, peer_devid = #{peer_devid}.")
        b.read_int == 1 ? true : false
    end


    # Enable the current device to access the memory of the peer device.
    # @param [Integer] peer_devid The peer device's ID.
    # @param [Integer] flags Currently flags must be set to zero.
    # @return [Class] This class.
    #
    # @since CUDA 4.0
    def self.enable_peer_access(peer_devid, flags = 0)
        status = API::cudaDeviceEnablePeerAccess(peer_devid, flags)
        Pvt::handle_error(status, "Failed to enable peer access: peer_devid = #{peer_devid}, flags = #{flags}.")
        self
    end


    # Disable the current device from accessing the memory of the peer device.
    # @param [Integer] peer_devid The peer device's ID.
    # @return [Class] This class.
    #
    # @since CUDA 4.0
    def self.disable_peer_access(peer_devid)
        status = API::cudaDeviceDisablePeerAccess(peer_devid)
        Pvt::handle_error(status, "Failed to disable peer access: peer_devid = #{peer_devid}.")
        self
    end

end

end # module
end # module
