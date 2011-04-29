#-----------------------------------------------------------------------
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
#-----------------------------------------------------------------------

require 'test/unit'
require_relative 'testbase'


class TestCUFunction < Test::Unit::TestCase

    include CUTestBase


    def test_function_set_param
        da = CUDevice.malloc(1024)
        db = CUDevice.malloc(1024)
        dc = CUDevice.malloc(1024)
        @func.param = []
        @func.param = [da, db, dc, 10]

        f = @func.param_setf(0, 2.5)
        assert_instance_of(CUFunction, f)
        f = @func.param_seti(4, 33)
        assert_instance_of(CUFunction, f)
        v = Buffer.new(:int, 1)
        f = @func.param_setv(0, v, 4)
        assert_instance_of(CUFunction, f)
        f = @func.param_set_size(8)
        assert_instance_of(CUFunction, f)

        da.free
        db.free
        dc.free
    end


    def test_function_set_block_shape
        @func.block_shape = 2
        @func.block_shape = [2, 3]
        @func.block_shape = 2, 3, 4
    end


    def test_function_set_shared_size
        @func.shared_size = 0
        @func.shared_size = 1024
    end


    def test_function_launch
        assert_function_launch(10) do |f, params|
            f.param = params
            f.block_shape = 10
            f.launch
        end
    end


    def test_function_launch_grid
        assert_function_launch(10) do |f, params|
            f.param = params
            f.block_shape = 10
            f.launch_grid(1)
        end

        assert_function_launch(20) do |f, params|
            f.param = params
            f.block_shape = 5
            f.launch_grid(4)
        end
    end


    def test_function_launch_kernel
        assert_function_launch(30) do |f, params|
            f.launch_kernel(3, 1, 1, 10, 1, 1, 0, 0, params)
        end
    end


    def test_function_launch_async
        assert_nothing_raised do
            assert_function_launch(10) do |f, params|
                f.param = params
                f.block_shape = 10
                f.launch_grid_async(1, 0)
            end

            assert_function_launch(20) do |f, params|
                f.param = params
                f.block_shape = 5
                f.launch_grid_async(4, 0)
            end
        end
    end


    def test_function_get_attribute
        CUFunctionAttribute.symbols.each do |k|
            v = @func.attribute(k)
            assert_instance_of(Fixnum, v)
        end
    end


    def test_function_set_cache_config
        CUFunctionCache.symbols.each do |k|
            @func.cache_config = k
        end
    end

private

    def assert_function_launch(size)
        type = :int
        element_size = 4
        nbytes = size*element_size
        da = CUDevice.malloc(nbytes)
        db = CUDevice.malloc(nbytes)
        dc = CUDevice.malloc(nbytes)
        ha = Buffer.new(type, size)
        hb = Buffer.new(type, size)
        hc = Array.new(size)
        hd = Buffer.new(type, size)
        (0...size).each { |i|
            ha[i] = i
            hb[i] = 1
            hc[i] = ha[i] + hb[i]
            hd[i] = 0
        }
        CUMemory.memcpy_htod(da, ha, nbytes)
        CUMemory.memcpy_htod(db, hb, nbytes)

        yield @func, [da, db, dc, size]

        CUMemory.memcpy_dtoh(hd, dc, nbytes)
        (0...size).each { |i|
            assert_equal(hc[i], hd[i])
        }
        da.free
        db.free
        dc.free
    end

end
