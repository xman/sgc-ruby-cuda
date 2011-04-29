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


class TestCUModule < Test::Unit::TestCase

    include CUTestBase


    def test_module_load_unload
        path = prepare_ptx
        m = CUModule.new
        m = m.load(path)
        assert_instance_of(CUModule, m)
        m = m.unload
        assert_instance_of(CUModule, m)
    end


    def test_module_load_with_invalid_ptx
        assert_raise(CUInvalidImageError) do
            m = CUModule.new
            m = m.load('test/bad.ptx')
        end

        assert_raise(CUFileNotFoundError) do
            m = CUModule.new
            m = m.load('test/notexists.ptx')
        end
    end


    def test_module_load_data
        path = prepare_ptx
        m = CUModule.new
        File.open(path) do |f|
            str = f.read
            r = m.load_data(str)
            assert_instance_of(CUModule, r)
            m.unload
        end
    end


    def test_module_load_data_with_invalid_ptx
        assert_raise(CUInvalidImageError) do
            m = CUModule.new
            str = "invalid ptx"
            m.load_data(str)
        end
    end


    def test_module_get_function
        f = @mod.function('vadd')
        assert_instance_of(CUFunction, f)
    end


    def test_module_get_function_with_invalid_name
        assert_raise(CUInvalidValueError) do
            f = @mod.function('')
        end

        assert_raise(CUReferenceNotFoundError) do
            f = @mod.function('badname')
        end
    end


    def test_module_get_global
        devptr, nbytes = @mod.global('gvar')
        assert_equal(4, nbytes)
        u = Buffer.new(:int, 1)
        CUMemory.memcpy_dtoh(u, devptr, nbytes)
        assert_equal(1997, u[0])
    end


    def test_module_get_global_with_invalid_name
        assert_raise(CUInvalidValueError) do
            devptr, nbytes = @mod.global('')
        end

        assert_raise(CUReferenceNotFoundError) do
            devptr, nbytes = @mod.global('badname')
        end
    end

end
