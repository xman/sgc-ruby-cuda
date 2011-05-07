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
require 'cuda/runtime/cuda'
require 'cuda/runtime/error'
require 'cuda/runtime/stream'
require 'memory/pointer'
require 'dl'


module SGC
module Cuda

class CudaFunction

    attr_reader :name


    # Create an instance to function _name_.
    def initialize(name)
        @name = name
    end


    # @return [CudaFunctionAttributes] The attributes of this kernel function.
    def attributes
        a = CudaFunctionAttributes.new
        status = API::cudaFuncGetAttributes(a.to_ptr, @name)
        Pvt::handle_error(status, "Failed to query function attributes.")
        a
    end


    # Set the preferred cache configuration to use for next launch on this kernel function.
    # @param [CudaFunctionCache] conf The preferred cache configuration.
    def cache_config=(conf)
        status = API::cudaFuncSetCacheConfig(@name, conf)
        Pvt::handle_error(status, "Failed to set function cache config: config = #{conf}.")
    end


    # Launch this kernel function with pre-configured settings.
    # @return [Class] This class.
    #
    # @see .configure
    def launch
        status = API::cudaLaunch(@name)
        Pvt::handle_error(status, "Failed to launch kernel function: name = #{@name}.")
        self
    end


    # Configure the settings for the next kernel launch.
    # @param [Dim3] grid_dim The 3D grid dimensions x, y, z to launch.
    # @param [Dim3] block_dim The 3D block dimensions x, y, z to launch.
    # @param [Integer] shared_mem_size Number of bytes of dynamic shared memory for each thread block.
    # @param [Integer, CudaStream] stream The stream to launch this kernel function on.
    #     Setting _stream_ to anything other than an instance of CudaStream will execute on the default stream 0.
    # @return [Class] This class.
    def self.configure(grid_dim, block_dim, shared_mem_size = 0, stream = 0)
        s = Pvt::parse_stream(stream)
        status = API::cudaConfigureCall(grid_dim.to_api, block_dim.to_api, shared_mem_size, s)
        Pvt::handle_error(status, "Failed to configure kernel function launch settings.\n" +
            "* #{grid_dim.x} x #{grid_dim.y} x #{grid_dim.z} grid\n" +
            "* #{block_dim.x} x #{block_dim.y} x #{block_dim.z} blocks\n" +
            "* shared memory size = #{shared_mem_size}")
        self
    end


    # Set the argument list of subsequent kernel function launch.
    # @param [Array] *args The list of arguments to pass to the kernel.
    # @return [Class] This class.
    def self.setup(*args)
        offset = 0
        args.each do |x|
            case x
                when Fixnum
                    p = FFI::MemoryPointer.new(:int).write_int(x)
                    size = 4
                when Float
                    p = FFI::MemoryPointer.new(:float).write_float(x)
                    size = 4
                when SGC::Memory::MemoryPointer
                    p = x.ref
                    size = FFI::TypeDefs[:pointer].size
                else
                    raise TypeError, "Invalid type of kernel parameters #{x}."
            end
            offset = align_up(offset, size)
            status = API::cudaSetupArgument(p, size, offset)
            Pvt::handle_error(status, "Failed to setup kernel argument for #{x}.")
            offset += size
        end
        self
    end


    # Load a dynamic library with _name_ from dynamic library path.
    # @param [String] name The name of the dynamic library to load.
    #     For library libcudart.so, its name is cudart.
    # @return [Class] This class.
    def self.load_lib(name)
        raise NotImplementedError
    end


    # Load a dynamic library from the given path.
    # @param [String] path The path of the dynamic library to load.
    # @return [Class] This class.
    def self.load_lib_file(path)
        @@libs << DL::dlopen(path)
        self
    end


    # Unload all the loaded dynamic libraries.
    # @return [Class] This class.
    def self.unload_all_libs
        @@libs.each do |h|
            h.close
        end
        @@libs = []
        self
    end


    @@libs = [] # @private

private

    def self.align_up(offset, alignment)
        (offset + alignment - 1) & ~(alignment - 1)
    end

end

end # module
end # module
