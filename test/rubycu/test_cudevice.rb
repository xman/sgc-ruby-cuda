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


class TestCUDevice < Test::Unit::TestCase

    include CUTestBase


    def test_device_count
        count = CUDevice.count
        assert(count > 0, "Device count failed.")
    end


    def test_device_query
        d = @dev
        assert_instance_of(CUDevice, d)
        assert_device_name(d)
        assert_device_memory_size(d)
        assert_device_capability(d)
        assert_device_properties(d)
        CUDeviceAttribute.symbols.each do |k|
            v = d.attribute(k)
            assert_instance_of(Fixnum, v)
        end
    end


    def test_device_malloc_free
        p = CUDevice.malloc(1024)
        assert_instance_of(CUDevicePtr, p)
        r = p.free
        assert_nil(r)
    end


    def test_device_malloc_with_huge_mem
        assert_raise(CUOutOfMemoryError) do
            size = @dev.total_mem + 1
            CUDevice.malloc(size)
        end
    end


    def test_device_can_access_peer
        current_dev = CUContext.device
        count = CUDevice.count
        (0...count).each do |devid|
            dev = CUDevice.get(devid)
            CUDevice.can_access_peer?(dev)
            CUDevice.can_access_peer?(current_dev, dev)
        end
    end

end
