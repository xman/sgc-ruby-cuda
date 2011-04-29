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
require 'memory/buffer'

include SGC::Memory


class TestMemoryBuffer < Test::Unit::TestCase

    def test_buffer_initialize
        b = Buffer.new(:int, 10)
        assert_instance_of(Buffer, b)
        assert_equal(10, b.size)

        b = Buffer.new(:long, 20)
        assert_instance_of(Buffer, b)
        assert_equal(20, b.size)

        b = Buffer.new(:float, 30)
        assert_instance_of(Buffer, b)
        assert_equal(30, b.size)
    end


    def test_buffer_element_size
        assert_equal(4, Buffer.element_size(:int))
        assert_equal(4, Buffer.element_size(:float))
    end


    def test_buffer_offset
        b = Buffer.new(:int, 16)
        c = b.offset(4)
        assert_kind_of(MemoryPointer, c)

        b = Buffer.new(:int, 10)
        c = b.offset(5)
        assert_kind_of(MemoryPointer, c)

        b = Buffer.new(:long, 10)
        c = b.offset(3)
        assert_kind_of(MemoryPointer, c)

        b = Buffer.new(:float, 10)
        c = b.offset(4)
        assert_kind_of(MemoryPointer, c)
    end


    def test_buffer_access
        b = Buffer.new(:int, 10)
        b[0] = 10
        assert_equal(10, b[0])
        b[9] = 20
        assert_equal(20, b[9])

        b = Buffer.new(:long, 10)
        b[3] = 2**40
        assert_equal(2**40, b[3])
        b[7] = 2**50
        assert_equal(2**50, b[7])

        b = Buffer.new(:float, 10)
        b[2] = 3.14
        assert_in_delta(3.14, b[2])
        b[8] = 9.33
        assert_in_delta(9.33, b[8])
    end

end
