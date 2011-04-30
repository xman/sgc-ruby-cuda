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


class TestCudaDevice < Test::Unit::TestCase

    include CudaTestBase


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
            CudaDevice.current = devid
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
            CudaDevice.flags = k
        end
    end


    def test_device_valid_devices
        count = CudaDevice.count
        devs = []
        (0...count).each do |devid|
            devs << devid
        end
        CudaDevice.valid_devices = devs
    end


    def test_device_cache_config
        if CudaDevice.properties.major >= 2
            CudaFunctionCache.symbols.each do |k|
                CudaDevice.cache_config = k
                c = CudaDevice.cache_config
                assert_equal(k, c)
            end
        end
    end


    def test_device_limit
        CudaLimit.symbols.each do |k|
            begin
                u = CudaDevice.limit(k)
                CudaDevice.limit = [k, u]
                v = CudaDevice.limit(k)
                assert_equal(u, v)
            rescue CudaUnsupportedLimitError
            end
        end
    end


    def test_device_reset
        r = CudaDevice.reset
        assert_equal(CudaDevice, r)
    end


    def test_device_synchronize
        r = CudaDevice.synchronize
        assert_equal(CudaDevice, r)
    end


    def test_device_can_access_peer
        current_devid = CudaDevice.get
        count = CudaDevice.count
        (0...count).each do |devid|
            CudaDevice.can_access_peer?(devid)
            CudaDevice.can_access_peer?(current_devid, devid)
        end
    end


    def test_device_enable_disable_peer_access
        current_devid = CudaDevice.get
        count = CudaDevice.count
        (0...count).each do |devid|
            if CudaDevice.can_access_peer?(devid)
                assert_nothing_raised do
                    CudaDevice.enable_peer_access(devid)
                    CudaDevice.disable_peer_access(devid)
                end
            else
                assert_raise(CudaInvalidDeviceError) do
                    CudaDevice.enable_peer_access(devid)
                end
                assert_raise(CudaPeerAccessNotEnabledError) do
                    CudaDevice.disable_peer_access(devid)
                end
            end
        end
    end

end
