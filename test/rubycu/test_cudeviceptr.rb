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


class TestCUDevicePtr < Test::Unit::TestCase

    include CUTestBase


    def test_device_ptr_offset
        devptr = CUDevice.malloc(1024)
        p = devptr.offset(1024)
        assert_instance_of(CUDevicePtr, p)
        p = devptr.offset(-1024)
        assert_instance_of(CUDevicePtr, p)
        devptr.free
    end


    def test_device_ptr_get_attribute
        if @dev.attribute(:UNIFIED_ADDRESSING) > 0
            devptr = CUDevice.malloc(4096)

            context = devptr.attribute(:CONTEXT)
            assert_kind_of(CUContext, context)

            mem_type = devptr.attribute(:MEMORY_TYPE)
            assert_equal(:DEVICE, memtype)

            dptr = devptr.attribute(:DEVICE_POINTER)
            assert_kind_of(CUDevicePtr, dptr)

            assert_raise(CUInvalidValueError) do
                devptr.attribute(:HOST_POINTER)
            end

            devptr.free

            CUPointerAttribute.symbols.each do |k|
                assert_raise(CUInvalidValueError) do
                    devptr.attribute(k)
                end
            end
        end
    end

end
