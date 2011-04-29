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


class TestCudaMemory < Test::Unit::TestCase

    include CudaTestBase


    def test_cuda_device_memory_malloc_free
        p = CudaDeviceMemory.malloc(1024)
        assert_instance_of(SGC::Memory::MemoryPointer, p)
        r = CudaDeviceMemory.free(p)
        assert_nil(r)

        p = CudaDeviceMemory.malloc(1024)
        assert_instance_of(SGC::Memory::MemoryPointer, p)
        r = p.free
        assert_nil(r)
    end


    def test_memory_copy
        size = 16
        type = :int
        nbytes = size*Buffer.element_size(type)

        a = Buffer.new(type, size)
        b = Buffer.new(type, size)
        c = Buffer.new(type, size)
        d = Buffer.new(type, size)
        p = CudaDeviceMemory.malloc(nbytes)
        q = CudaDeviceMemory.malloc(nbytes)

        (0...size).each do |i|
            a[i] = 0
            b[i] = 0
            c[i] = 0
            d[i] = 0
        end

        extend CudaMemory

        memcpy_htoh(b, a, nbytes)
        (0...size).each do |i|
            assert_equal(a[i], b[i])
        end

        memcpy_htod(p, b, nbytes)
        memcpy_dtoh(c, p, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], c[i])
        end

        memcpy_dtod(q, p, nbytes)
        CudaThread.synchronize
        memcpy_dtoh(d, q, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], d[i])
        end

        CudaDeviceMemory.free(p)
        CudaDeviceMemory.free(q)
    end

end
