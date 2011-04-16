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


module SGC
module Memory

# A memory pointer class.
class MemoryPointer

    # @param [Integer] addr Memory address _addr_ to initialize to.
    # @return A memory pointer pointing to address _addr_.
    def initialize(addr = nil)
        @p = FFI::MemoryPointer.new(:pointer)
        @p.write_long(addr.to_i)
    end


    # @return The internal pointer representation.
    def ptr
        @p.read_pointer
    end


    # Set this pointer to point to memory address _addr_.
    # @param [Integer] addr Memory address to set to.
    # @return _addr_.
    def ptr=(addr)
        @p.write_pointer(addr)
        addr
    end


    # @param [Integer] index Index to a memory offset.
    # @return [MemoryPointer] A memory pointer pointing to the _index_ byte.
    def offset(index)
        MemoryPointer.new(@p.read_pointer.to_i + index)
    end


    # @return The internal representation of a pointer pointing to this memory pointer.
    def ref
        @p
    end

end

end # module
end # module
