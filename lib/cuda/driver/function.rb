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

class CUFunction

    # @deprecated Use {#launch_kernel}.
    #
    # Set the argument list of subsequent function call to _arg1_, _arg2_, *other_args.
    # @param *args The list of arguments to pass to the kernel function.
    def param=(*args)
        offset = 0
        args.flatten.each do |x|
            case x
                when Fixnum
                    p = FFI::MemoryPointer.new(:int)
                    p.write_int(x)
                    size = 4
                when Float
                    p = FFI::MemoryPointer.new(:float)
                    p.write_float(x)
                    size = 4
                when CUDevicePtr
                    p = FFI::MemoryPointer.new(:CUDevicePtr)
                    API::write_cudeviceptr(p, x.to_api)
                    size = p.size
                else
                    raise TypeError, "Invalid type of argument #{x.to_s}."
            end
            offset = align_up(offset, size)
            status = API::cuParamSetv(self.to_api, offset, p, size)
            Pvt::handle_error(status, "Failed to set function parameters: offset = #{offset}, value = #{x}.")
            offset += size
        end

        status = API::cuParamSetSize(self.to_api, offset)
        Pvt::handle_error(status, "Failed to set function parameter size: size = #{offset}.")
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Set a float parameter to the function's argument list at _offset_ with _value_.
    # @param [Integer] offset Number of bytes to offset.
    # @param [Float] value The floating-point value to set as the function parameter.
    # @return [CUFunction] This function.
    def param_setf(offset, value)
        status = API::cuParamSetf(self.to_api, offset, value)
        Pvt::handle_error(status, "Failed to set function float parameter: offset = #{offset}, value = #{value}.")
        self
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Set an integer parameter to the function's argument list at _offset_ with _value_.
    # @param [Integer] offset Number of bytes to offset.
    # @param [Integer] value The integer value to set as the function parameter.
    # @return [CUFunction] This function.
    def param_seti(offset, value)
        status = API::cuParamSeti(self.to_api, offset, value)
        Pvt::handle_error(status, "Failed to set function integer parameter: offset = #{offset}, value = #{value}")
        self
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Set an arbitrary data to the function's argument list at _offset_ with _ptr_ pointed _nbytes_ data.
    # @param [Integer] offset Number of bytes to offset.
    # @param [CUDevicePtr] ptr A device pointer pointing to an arbitrary data to be used as the function parameter.
    # @param [Integer] nbytes The size of the arbitrary data in bytes.
    # @return [CUFunction] This function.
    def param_setv(offset, ptr, nbytes)
        p = FFI::MemoryPointer.new(:pointer)
        API::write_size_t(p, ptr.to_api.to_i) # Workaround broken p.write_pointer() on 64bit pointer.
        status = API::cuParamSetv(self.to_api, offset, p, nbytes)
        Pvt::handle_error(status, "Failed to set function arbitrary parameter: offset = #{offset}, size = #{nbytes}.")
        self
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Set the function parameter size to _nbytes_.
    # @param [Integer] nbytes The parameter size in bytes.
    # @return [CUFunction] This function.
    def param_set_size(nbytes)
        status = API::cuParamSetSize(self.to_api, nbytes)
        Pvt::handle_error(status, "Failed to set function parameter size: size = #{nbytes}.")
        self
    end


    # @deprecated
    #
    def param_set_texref(texunit, texref)
        raise NotImplementedError
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Set the block dimensions to use for next launch.
    # @overload block_shape=(xdim)
    # @overload block_shape=(xdim, ydim)
    # @overload block_shape=(xdim, ydim, zdim)
    # @param [Integer] xdim The size of the x dimension.
    # @param [Integer] ydim The size of the y dimension. Defaults to 1.
    # @param [Integer] zdim The size of the z dimension. Defaults to 1.
    def block_shape=(*args)
        xdim, ydim, zdim = args.flatten
        ydim = 1 if ydim.nil?
        zdim = 1 if zdim.nil?
        status = API::cuFuncSetBlockShape(self.to_api, xdim, ydim, zdim)
        Pvt::handle_error(status, "Failed to set function block shape: (x,y,z) = (#{xdim},#{ydim},#{zdim}).")
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Set the dynamic shared-memory size to use for next launch.
    # @param [Integer] nbytes Number of bytes.
    def shared_size=(nbytes)
        status = API::cuFuncSetSharedSize(self.to_api, nbytes)
        Pvt::handle_error(status, "Failed to set function shared memory size: #{nbytes}.")
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Launch this kernel function with 1x1x1 grid of blocks to execute on the current CUDA device.
    # @return [CUFunction] This function.
    def launch
        status = API::cuLaunch(self.to_api)
        Pvt::handle_error(status, "Failed to launch kernel function on 1x1x1 grid of blocks.")
        self
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Launch this kernel function with grid dimensions (_xdim_, _ydim_) to execute on the current CUDA device.
    # @overload launch_grid(xdim)
    # @overload launch_grid(xdim, ydim)
    # @param [Integer] xdim The x dimensional size of the grid to launch.
    # @param [Integer] ydim The y dimensional size of the grid to launch. Defaults to 1.
    # @return [CUFunction] This function.
    def launch_grid(xdim, ydim = 1)
        status = API::cuLaunchGrid(self.to_api, xdim, ydim)
        Pvt::handle_error(status, "Failed to launch kernel function on #{xdim}x#{ydim} grid of blocks.")
        self
    end


    # @deprecated Use {#launch_kernel}.
    #
    # Launch this kernel function with grid dimensions (_xdim_, _ydim_) on _stream_ asynchronously to execute
    # on the current CUDA device. Setting _stream_ to anything other than an instance of CUStream
    # will execute on the default stream 0.
    # @overload launch_grid_async(xdim, stream)
    # @overload launch_grid_async(xdim, ydim, stream)
    # @param [Integer] xdim The x dimensional size 
    def launch_grid_async(xdim, ydim = 1, stream)
        s = Pvt::parse_stream(stream)
        status = API::cuLaunchGridAsync(self.to_api, xdim, ydim, s)
        Pvt::handle_error(status, "Failed to launch kernel function asynchronously on #{xdim}x#{ydim} grid of blocks.")
        self
    end


    # @param [CUFunctionAttribute] attrib The attribute of the kernel function to query.
    # @return [Integer] The particular attribute _attrib_ of this kernel function.
    #
    # @example Get function attribute.
    #    func.attribute(:MAX_THREADS_PER_BLOCK)    #=> 512
    #    func.attribute(:SHARED_SIZE_BYTES)        #=> 44
    #    func.attribute(:NUM_REGS)                 #=> 3
    def attribute(attrib)
        p = FFI::MemoryPointer.new(:int)
        status = API::cuFuncGetAttribute(p, attrib, self.to_api)
        Pvt::handle_error(status, "Failed to query function attribute: attribute = #{attrib}.")
        p.read_int
    end


    # Set the preferred cache configuration (CUFunctionCache) to use for next launch.
    # @param [CUFunctionCache] conf The preferred cache configuration.
    def cache_config=(conf)
        status = API::cuFuncSetCacheConfig(self.to_api, conf)
        Pvt::handle_error(status, "Failed to set function cache config: config = #{conf}.")
    end

    # Launch this kernel function with full configuration parameters and function parameters
    # to execute on the current CUDA device.
    # @param [Integer] grid_xdim The x dimensional size of the grid to launch.
    # @param [Integer] grid_ydim The y dimensional size of the grid to launch.
    # @param [Integer] grid_zdim The z dimensional size of the grid to launch.
    # @param [Integer] block_xdim The x dimensional size of a block in the grid.
    # @param [Integer] block_ydim The y dimensional size of a block in the grid.
    # @param [Integer] block_zdim The z dimensional size of a block in the grid.
    # @param [Integer] shared_mem_size Number of bytes of dynamic shared memory for each thread block.
    # @param [Integer, CUStream] stream The stream to launch this kernel function on.
    #     Setting _stream_ to anything other than an instance of CUStream will execute on the default stream 0.
    # @param [Array<Fixnum, Float, CUDevicePtr>] params The list of parameters to pass in for the kernel function launch.
    #     * A Fixnum is mapped to a C type int.
    #     * A Float is mapped to a C type float.
    # @return [CUFunction] This function.
    #
    # @todo Add support for other C data types for the kernel function parameters.
    def launch_kernel(grid_xdim, grid_ydim, grid_zdim, block_xdim, block_ydim, block_zdim, shared_mem_size, stream, params)
        p = parse_params(params)
        s = Pvt::parse_stream(stream)
        status = API::cuLaunchKernel(self.to_api, grid_xdim, grid_ydim, grid_zdim, block_xdim, block_ydim, block_zdim, shared_mem_size, s, p, nil)
        Pvt::handle_error(status, "Failed to launch kernel function.\n" +
            "* #{grid_xdim} x #{grid_ydim} x #{grid_zdim} grid\n" +
            "* #{block_xdim} x #{block_ydim} x #{block_zdim} blocks\n" +
            "* shared memory size = #{shared_mem_size}")
        self
    end


    # @private
    def initialize(ptr)
        @pfunc = ptr
    end
    private_class_method :new


    # @private
    def to_api
        API::read_cufunction(@pfunc)
    end

private

    def parse_params(params)
        params.is_a?(Array) or raise TypeError, "Expect _params_ an Array, but we get a #{params.class}."
        params.size <= 0 and return nil

        p = FFI::MemoryPointer.new(:pointer, params.size)
        params.each_with_index do |x,i|
            m = case x
                when Fixnum
                    FFI::MemoryPointer.new(:int).write_int(x)
                when Float
                    FFI::MemoryPointer.new(:float).write_float(x)
                when CUDevicePtr
                    ptr = FFI::MemoryPointer.new(:CUDevicePtr)
                    API::write_cudeviceptr(ptr, x.to_api)
                    ptr
                else
                    raise TypeError, "Invalid type of kernel parameter #{x.to_s}."
            end
            p[i].write_pointer(m)
        end
        p
    end


    def align_up(offset, alignment)
        (offset + alignment - 1) & ~(alignment - 1)
    end

end

end # module
end # module
