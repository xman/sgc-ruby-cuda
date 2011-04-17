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

    def self.count
        p = FFI::MemoryPointer.new(:int)
        status = API::cudaGetDeviceCount(p)
        Pvt::handle_error(status)
        p.read_int
    end


    def self.get
        p = FFI::MemoryPointer.new(:int)
        status = API::cudaGetDevice(p)
        Pvt::handle_error(status)
        p.read_int
    end

    def self.current; self.get; end


    def self.set(devid)
        status = API::cudaSetDevice(devid)
        Pvt::handle_error(status)
        self
    end
    class << self; alias_method :current=, :set; end


    def self.choose(prop)
        pdev = FFI::MemoryPointer.new(:int)
        status = API::cudaChooseDevice(pdev, prop.to_ptr)
        Pvt::handle_error(status)
        pdev.read_int
    end


    def self.properties(devid = self.get)
        prop = API::CudaDeviceProp.new
        status = API::cudaGetDeviceProperties(prop.to_ptr, devid)
        Pvt::handle_error(status)
        prop
    end


    def self.flags=(flags)
        if flags.is_a?(Symbol)
            flags = CudaDeviceFlags[flags]
        end

        status = API::cudaSetDeviceFlags(flags)
        Pvt::handle_error(status)
        flags
    end


    def self.valid_devices=(devs)
        p = FFI::MemoryPointer.new(:int, devs.count)
        devs.each_with_index do |devid, i|
            p[i].write_int(devid)
        end
        status = API::cudaSetValidDevices(p, devs.count)
        Pvt::handle_error(status)
        devs
    end

end

end # module
end # module
