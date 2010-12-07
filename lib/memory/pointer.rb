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
require 'memory/interface/ipointer'


module SGC
module Memory

class MemoryPointer

    include IMemoryPointer


    def initialize(v = nil)
        @p = FFI::MemoryPointer.new(:pointer)
        @p.write_pointer(v)
    end


    def ptr
        @p.read_pointer
    end


    def ptr=(v)
        @p.write_pointer(v)
        v
    end


    def offset(i)
        MemoryPointer.new(@p.read_pointer.to_i + i)
    end


    def ref
        @p
    end

end

end # module
end # module
