#
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
#

require 'ffi'

require 'memory/interface/ibuffer'
require 'memory/pointer'


module SGC
module Memory

# A memory buffer class which implements {IBuffer} interface.
# @see IBuffer
# @see IBuffer::ClassMethods
class Buffer

    include IBuffer


    # @param [Symbol] type A symbol corresponds to a supported C data type, e.g. :int, :long, :float.
    # @param [Integer] size The number of elements.
    # @return A buffer with _size_ elements of _type_.
    def initialize(type, size)
        @@reads[type] && @@writes[type] or raise "Invalid buffer element type."

        @reader = @@reads[type]
        @writer = @@writes[type]
        @ptr = FFI::MemoryPointer.new(type, size)
        @size = size
    end


    # @param [Integer] index The index (0..size-1) of the element to return.
    # @return The element at _index_ of this buffer.
    def [](index)
        assert_index(index)
        @ptr[index].send(@reader)
    end


    # Set the element at _index_ of this buffer to _value_.
    # @param [Integer] index The index (0..size-1) of the element to set.
    # @param [Object] value The value to set to.
    # @return _value_.
    def []=(index, value)
        assert_index(index)
        @ptr[index].send(@writer, value)
        value
    end


    # @return [Integer] The number of elements in this buffer.
    def size
        @size
    end


    # @return [Integer] The size of an element in this buffer in bytes.
    def element_size
        @ptr.type_size
    end


    # @private
    def ptr
        @ptr
    end


    # @private
    def to_api
        @ptr
    end


    # @param [Integer] index The index to an element in this buffer.
    # @return [MemoryPointer] A memory pointer pointing to the _index_ element.
    def offset(index)
        assert_index(index)
        MemoryPointer.new(@ptr[index])
    end


    # @param [Symbol] type A symbol corresponds to a supported C data type, e.g. :int, :long, :float.
    # @return [Integer] The size of an element of _type_.
    def self.element_size(type)
        @@sizes[type]
    end

protected

    def assert_index(i)
        i >= 0 && i < @size or raise IndexError, "Invalid index to buffer: index = #{i}. Expect index in 0..#{@size-1}"
    end

    @@reads = { int: :read_int, long: :read_long, float: :read_float } # @private
    @@writes = { int: :write_int, long: :write_long, float: :write_float } # @private
    @@sizes = { int: 4, long: FFI::TypeDefs[:long].size, float: 4 } # @private

end

end # module
end # module
