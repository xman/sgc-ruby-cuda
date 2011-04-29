#-----------------------------------------------------------------------
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
#-----------------------------------------------------------------------

require 'test/unit'
require_relative 'testbase'


class TestCudaFunction < Test::Unit::TestCase

    include CudaTestBase


    def test_function_name
        f = CudaFunction.new("vadd")
        assert_equal("vadd", f.name)
    end


    def test_function_attributes
        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)
        f = CudaFunction.new("vadd")
        a = f.attributes
        assert_instance_of(CudaFunctionAttributes, a)
        CudaFunction.unload_all_libs
    end


    def test_function_cache_config
        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)
        f = CudaFunction.new("vadd")
        CudaFunctionCache.symbols.each do |k|
            f.cache_config = k
        end
        CudaFunction.unload_all_libs
    end


    def test_function_launch
        type = :int
        size = 10
        nbytes = size*4

        a = Buffer.new(type, size)
        b = Buffer.new(type, size)
        c = Buffer.new(type, size)

        p = CudaDeviceMemory.malloc(nbytes)
        q = CudaDeviceMemory.malloc(nbytes)
        r = CudaDeviceMemory.malloc(nbytes)

        (0...size).each do |i|
            a[i] = i
            b[i] = 2
            c[i] = 0
        end

        CudaMemory.memcpy_htod(p, a, nbytes)
        CudaMemory.memcpy_htod(q, b, nbytes)
        CudaMemory.memcpy_htod(r, c, nbytes)

        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)

        CudaFunction.configure(Dim3.new(1, 1, 1), Dim3.new(size, 1, 1))
        CudaFunction.setup(p, q, r, size)

        f = CudaFunction.new("vadd")
        f.launch

        CudaMemory.memcpy_dtoh(c, r, nbytes)

        (0...size).each do |i|
            assert_equal(a[i] + b[i], c[i])
        end

        CudaFunction.unload_all_libs

        CudaDeviceMemory.free(p)
        CudaDeviceMemory.free(q)
        CudaDeviceMemory.free(r)
    end

end
