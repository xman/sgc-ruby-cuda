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

require 'memory/interface/ipointer'


module SGC
module Memory

module IBuffer

    include IMemoryPointer

    def initialize(type, size); end

    def [](index); raise NotImplementedError; end
    def []=(index, value); raise NotImplementedError; end
    def size; raise NotImplementedError; end
    def element_size; raise NotImplementedError; end

    module ClassMethods
        def element_size(type); raise NotImplementedError; end
    end

    def self.included(base)
        base.extend(ClassMethods)
    end

end

end # module
end # module
