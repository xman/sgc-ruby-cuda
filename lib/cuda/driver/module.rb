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
require 'cuda/driver/function'


module SGC
module CU

class CUModule

    # Allocate a CUDA module.
    def initialize
        @pmod = FFI::MemoryPointer.new(:CUModule)
    end


    # Load a compute module from the file at _path_ into the current CUDA context.
    # The file should be a cubin file or a PTX file.
    #
    # A PTX file may be obtained by compiling the .cu file using nvcc with -ptx option.
    #    $ nvcc -ptx vadd.cu
    #
    # @param [String] path The path of the file to load.
    # @return [CUModule] This CUDA module.
    def load(path)
        status = API::cuModuleLoad(@pmod, path)
        Pvt::handle_error(status, "Failed to load module: path = #{path}.")
        self
    end


    # Load a compute module from the String _image_str_ into the current CUDA context.
    # @param [String] image_str A string which contains a cubin or a PTX data.
    # @return [CUModule] This CUDA module.
    #
    # @see #load
    def load_data(image_str)
        status = API::cuModuleLoadData(@pmod, image_str)
        Pvt::handle_error(status, "Failed to load module data.")
        self
    end


    # @note Not implemented yet.
    #
    # @see #load
    def load_data_ex
        raise NotImplementedError
    end


    # @note Not implemented yet.
    #
    # @see #load
    def load_fat_binary
        raise NotImplementedError
    end


    # Unload this CUDA module from the current CUDA context.
    # @return [CUModule] This CUDA module.
    def unload
        status = API::cuModuleUnload(self.to_api)
        Pvt::handle_error(status, "Failed to unload module.")
        self
    end


    # Lookup for a CUDA function corresponds to the function name _name_ in the loaded compute module.
    # A compute module was loaded with {CUModule#load} and alike methods.
    # @param [String] name The name of the function to lookup in the loaded compute modules.
    # @return [CUFunction] The CUDA function corresponds to _name_ in the loaded compute module.
    def function(name)
        p = FFI::MemoryPointer.new(:CUFunction)
        status = API::cuModuleGetFunction(p, self.to_api, name)
        Pvt::handle_error(status, "Failed to get module function: name = #{name}.")
        CUFunction.send(:new, p)
    end


    # Lookup for the device pointer and the size of the global variable _name_ in the loaded compute modules.
    # @param [String] name The name of the global variable to lookup in the loaded compute modules.
    # @return [Array(CUDevicePtr, Integer)] An array with a device pointer to the global variable and its size in bytes.
    def global(name)
        pdevptr = FFI::MemoryPointer.new(:CUDevicePtr)
        psize = FFI::MemoryPointer.new(:size_t)
        status = API::cuModuleGetGlobal(pdevptr, psize, self.to_api, name)
        Pvt::handle_error(status, "Failed to get module global: name = #{name}.")
        [CUDevicePtr.send(:new, pdevptr), API::read_size_t(psize)]
    end


    # @return A texture reference corresponds to the texture _name_ in the loaded compute module.
    #
    # @note Not implemented yet.
    def texref(name)
        raise NotImplementedError
    end


    # @return A surface texture reference corresponds to the surface _name_ in the loaded compute module.
    #
    # @note Not implemented yet.
    def surfref(name)
        raise NotImplementedError
    end


    # @private
    def to_api
        API::read_cufunction(@pmod)
    end

end

end # module
end # module
