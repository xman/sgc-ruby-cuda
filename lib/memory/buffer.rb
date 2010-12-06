#
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
#

require 'ffi'

require 'memory/interface/ibuffer'
require 'memory/pointer'


module SGC
module Memory

class Buffer

    include IBuffer


    def initialize(type, size)
        @@reads[type] && @@writes[type] or raise "Invalid buffer element type."

        @reader = @@reads[type]
        @writer = @@writes[type]
        @ptr = FFI::MemoryPointer.new(type, size)
        @size = size
    end


    def [](i)
        assert_index(i)
        @ptr[i].send(@reader)
    end


    def []=(i, v)
        assert_index(i)
        @ptr[i].send(@writer, v)
        v
    end


    def size
        @size
    end


    def element_size
        @ptr.type_size
    end


    def ptr
        @ptr
    end


    def offset(i)
        assert_index(i)
        MemoryPointer.new(@ptr[i])
    end


    def self.element_size(type)
        @@sizes[type]
    end

protected

    def assert_index(i)
        i >= 0 && i < @size or raise IndexError, "Invalid index to buffer: index = #{i}. Expect index in 0..#{@size-1}"
    end


    @@reads = { int: :read_int, long: :read_long, float: :read_float }
    @@writes = { int: :write_int, long: :write_long, float: :write_float }
    @@sizes = { int: 4, long: FFI::TypeDefs[:long].size, float: 4 }

end

end # module
end # module
