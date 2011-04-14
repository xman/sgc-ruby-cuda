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

module SGC
module Memory

# @abstract A memory buffer interface.
# A buffer stores elements of the same C data type.
module IBuffer

    # @param [Integer] index The index (0..size-1) of the element to return.
    # @return The element at _index_ of this buffer.
    def [](index); raise NotImplementedError; end

    # Set the element at _index_ of this buffer to _value_.
    # @param [Integer] index The index (0..size-1) of the element to set.
    # @param [Object] value The value to set to.
    # @return _value_.
    def []=(index, value); raise NotImplementedError; end

    # @return [Integer] The number of elements in this buffer.
    def size; raise NotImplementedError; end

    # @return [Integer] The size of an element in this buffer in bytes.
    def element_size; raise NotImplementedError; end

    # A set of methods automatically extended by the classes which _include_ _IBuffer_.
    module ClassMethods
        # @param [Symbol] type A symbol corresponds to a supported C data type, e.g. :int, :long, :float.
        # @return [Integer] The size of an element of _type_.
        def element_size(type); raise NotImplementedError; end
    end

    # @private
    def self.included(base)
        base.extend(ClassMethods)
    end

end

end # module
end # module
