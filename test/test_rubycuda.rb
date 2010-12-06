#-----------------------------------------------------------------------
# Copyright (c) 2010 Chung Shin Yee
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
require 'rubycuda'

include SGC::Cuda

DEVID = ENV['DEVID'].to_i


class TestRubyCuda < Test::Unit::TestCase

    def setup
        CudaDevice.current = DEVID
    end

    def teardown
        CudaThread.exit
    end


    def test_error
        CudaError.symbols.each do |k|
            s = get_error_string(k)
            assert(s.size > 0)
        end
        e = get_last_error
        s = get_error_string(e)
        assert(s.size > 0)

        e = peek_at_last_error
        s = get_error_string(e)
        assert(s.size > 0)
    end


    def test_version
        dv = driver_version
        rv = runtime_version
        assert(dv > 0)
        assert(rv > 0)
    end


    def test_device_count
        count = CudaDevice.count
        assert(count > 0, "Device count failed.")
    end

    def test_device_get_set
        count = CudaDevice.count
        (0...count).each do |devid|
            r = CudaDevice.set(devid)
            assert_equal(CudaDevice, r)
            d = CudaDevice.get
            assert_equal(devid, d)
        end

        count = CudaDevice.count
        (0...count).each do |devid|
            r = CudaDevice.current = devid
            assert_equal(devid, r)
            d = CudaDevice.current
            assert_equal(devid, d)
        end
    end

    def test_device_choose
        count = CudaDevice.count
        prop = CudaDeviceProp.new
        devid = CudaDevice.choose(prop)
        assert(devid >= 0 && devid < count)
    end

    def test_device_properties
        prop = CudaDevice.properties
        assert_instance_of(CudaDeviceProp, prop)
        # TODO: assert the content of the _prop_.
    end

    def test_device_flags
        CudaDeviceFlags.symbols.each do |k|
            r = CudaDevice.flags = k
            assert_equal(k, r)
            r = CudaDevice.flags = CudaDeviceFlags[k]
            assert_equal(CudaDeviceFlags[k], r)
        end
    end

    def test_device_valid_devices
        count = CudaDevice.count
        devs = []
        (0...count).each do |devid|
            devs << devid
        end
        r = CudaDevice.valid_devices = devs
        assert_equal(devs, r)
    end

    def test_thread_exit
        r = CudaThread.exit
        assert_equal(CudaThread, r)
    end

    def test_thread_cache_config
        CudaFuncCache.symbols.each do |k|
            r = CudaThread.cache_config = k
            assert_equal(k, r)
            c = CudaThread.cache_config
            assert_equal(k, c)
        end
    end

    def test_thread_limit
        CudaLimit.symbols.each do |k|
            v = CudaThread.limit(k)
            assert_kind_of(Integer, v)
            r = CudaThread.limit = [k, v]
            assert_equal([k, v], r)
        end
    end

    def test_thread_synchronize
        r = CudaThread.synchronize
        assert_equal(CudaThread, r)
    end

    def test_buffer_initialize
        bint = Buffer.new(:int, 16)
        assert_instance_of(Buffer, bint)
        assert_equal(16, bint.size)
        assert_equal(4, bint.element_size)
        assert_equal(4, Buffer.element_size(:int))

        blong = Buffer.new(:long, 10)
        assert_instance_of(Buffer, blong)
        assert_equal(10, blong.size)
        # TODO: Detect if this is 32bit or 64bit OS and check accordingly.
        assert(blong.element_size == 4 || blong.element_size == 8)
        assert(Buffer.element_size(:long) == 4 || Buffer.element_size(:long) == 8)

        bfloat = Buffer.new(:float, 20)
        assert_instance_of(Buffer, bfloat)
        assert_equal(20, bfloat.size)
        assert_equal(4, bfloat.element_size)
        assert_equal(4, Buffer.element_size(:float))
    end

    def test_buffer_access
        b = Buffer.new(:int, 16)
        b[0] = 10
        assert_equal(10, b[0])
        b[9] = 20
        assert_equal(20, b[9])

        b = Buffer.new(:float, 10)
        b[2] = 3.14
        assert_in_delta(3.14, b[2])
        b[8] = 9.33
        assert_in_delta(9.33, b[8])
    end

    def test_buffer_ptr
        b = Buffer.new(:int, 32)
        p = b.ptr
        assert_not_nil(p)
    end

    def test_buffer_offset
        size = 16
        b = Buffer.new(:int, size)
        (0...size).each do |i|
            b[i] = i
        end

        m0 = b.offset(0)
        m8 = b.offset(8)
        assert_equal(m0.ptr, m8.offset(-8*b.element_size).ptr)
    end

end
