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


class TestCudaError < Test::Unit::TestCase

    include CudaTestBase


    def test_error
        CudaError.symbols.each do |k|
            s = get_error_string(k)
            assert(s.size > 0)
        end
        e = get_last_error
        s = get_error_string(e)
        assert(s.size > 0)

        e = peek_at_last_error
        s = get_error_string(e)
        assert(s.size > 0)
    end

end
