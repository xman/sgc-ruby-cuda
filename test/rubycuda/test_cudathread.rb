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


class TestCudaThread < Test::Unit::TestCase

    include CudaTestBase


    def test_thread_exit
        r = CudaThread.exit
        assert_equal(CudaThread, r)
    end


    def test_thread_cache_config
        if CudaDevice.properties.major >= 2
            CudaFunctionCache.symbols.each do |k|
                CudaThread.cache_config = k
                c = CudaThread.cache_config
                assert_equal(k, c)
            end
        end
    end


    def test_thread_limit
        CudaLimit.symbols.each do |k|
            begin
                u = CudaThread.limit(k)
                CudaThread.limit = [k, u]
                v = CudaThread.limit(k)
                assert_equal(u, v)
            rescue CudaUnsupportedLimitError
            end
        end
    end


    def test_thread_synchronize
        r = CudaThread.synchronize
        assert_equal(CudaThread, r)
    end

end
