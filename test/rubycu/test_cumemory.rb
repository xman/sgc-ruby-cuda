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


class TestCUMemory < Test::Unit::TestCase

    include CUTestBase


    def test_memory_copy
        type = :int
        size = 16
        element_size = 4
        nbytes = size*element_size
        b = Buffer.new(type, size)
        c = Buffer.new(type, size)
        d = Buffer.new(type, size)
        e = Buffer.new(type, size)
        p = CUDevice.malloc(nbytes)
        q = CUDevice.malloc(nbytes)
        r = CUDevice.malloc(nbytes)

        (0...size).each do |i|
            b[i] = i
            c[i] = 0
            d[i] = 0
            e[i] = 0
        end

        CUMemory.memcpy_htod(p, b, nbytes)
        CUMemory.memcpy_dtoh(c, p, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], c[i])
        end

        (0...size).each do |i|
            b[i] = 2*i
            c[i] = 0
        end
        CUMemory.memcpy_htod_async(p, b, nbytes, 0)
        CUMemory.memcpy_dtoh_async(c, p, nbytes, 0)
        CUContext.synchronize
        (0...size).each do |i|
            assert_equal(b[i], c[i])
        end

        CUMemory.memcpy_dtod(q, p, nbytes)
        CUContext.synchronize
        CUMemory.memcpy_dtoh(d, q, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], d[i])
        end

        CUMemory.memcpy_dtod_async(r, p, size*element_size, 0)
        CUContext.synchronize
        CUMemory.memcpy_dtoh(e, r, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], e[i])
        end

        if false    # FIXME: The memcpy is not working.
        if @dev.attribute(:UNIFIED_ADDRESSING) > 0
            (0...size).each do |i|
                b[i] = i
                c[i] = 0
                d[i] = 0
                e[i] = 0
            end
            CUMemory.memcpy(p, b, nbytes)
            CUMemory.memcpy(c, p, nbytes)
            (0...size).each do |i|
                assert_equal(b[i], c[i])
            end
        end
        end

        p.free
        q.free
        r.free
    end


    def test_memory_get_info
        info = CUMemory.mem_info
        assert(info[:free] >= 0)
        assert(info[:total] >= 0)
    end

end
