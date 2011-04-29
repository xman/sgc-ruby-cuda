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


class TestCUContext < Test::Unit::TestCase

    include CUTestBase


    def test_context_create_destroy
        c = CUContext.create(@dev)
        assert_instance_of(CUContext, c)
        c = c.destroy
        assert_nil(c)
        c = CUContext.create(0, @dev)
        assert_instance_of(CUContext, c)
        c = c.destroy
        assert_nil(c)
        CUContextFlags.symbols.each do |k|
            c = CUContext.create(k, @dev)
            assert_instance_of(CUContext, c)
            c = c.destroy
            assert_nil(c)
        end
    end


    def test_context_current
        c = CUContext.current
        assert_instance_of(CUContext, c)
        CUContext.current = c
    end


    def test_context_attach_detach
        c1 = @ctx.attach(0)
        assert_instance_of(CUContext, c1)
        c2 = @ctx.detach
        assert_nil(c2)
    end


    def test_context_attach_nonzero_flags_detach
        assert_raise(CUInvalidValueError) do
            c1 = @ctx.attach(999)
            assert_instance_of(CUContext, c1)
            c2 = @ctx.detach
            assert_nil(c2)
        end
    end


    def test_context_push_pop_current
        c1 = CUContext.pop_current
        assert_instance_of(CUContext, c1)
        c2 = @ctx.push_current
        assert_instance_of(CUContext, c2)
    end


    def test_context_get_device
        d = CUContext.device
        assert_device(d)
    end


    def test_context_get_set_limit
        if @dev.compute_capability[:major] >= 2
            assert_limit = Proc.new { |&b| assert_nothing_raised(&b) }
        else
            assert_limit = Proc.new { |&b| assert_raise(CUUnsupportedLimitError, &b) }
        end
        assert_limit.call do
            stack_size = CUContext.limit(:STACK_SIZE)
            assert_kind_of(Integer, stack_size)
            fifo_size = CUContext.limit(:PRINTF_FIFO_SIZE)
            assert_kind_of(Integer, fifo_size)
            heap_size = CUContext.limit(:MALLOC_HEAP_SIZE)
            assert_kind_of(Integer, heap_size)
            CUContext.limit = [:STACK_SIZE, stack_size]
            s = CUContext.limit(:STACK_SIZE)
            assert_equal(stack_size, s)
            CUContext.limit = [:PRINTF_FIFO_SIZE, fifo_size]
            s = CUContext.limit(:PRINTF_FIFO_SIZE)
            assert_equal(fifo_size, s)
            CUContext.limit = :MALLOC_HEAP_SIZE, heap_size
            s = CUContext.limit(:MALLOC_HEAP_SIZE)
            assert_equal(heap_size, s)
        end
    end


    def test_context_get_set_cache_config
        if @dev.compute_capability[:major] >= 2
            config = CUContext.cache_config
            assert_not_nil(CUFunctionCache[config])
            CUContext.cache_config = config
            c = CUContext.cache_config
            assert_equal(config, c)
        else
            config = CUContext.cache_config
            assert_equal(:PREFER_NONE, config)
            CUContext.cache_config = config
            c = CUContext.cache_config
            assert_equal(:PREFER_NONE, c)
        end
    end


    def test_context_get_api_version
        v1 = @ctx.api_version
        v2 = CUContext.api_version
        assert_kind_of(Integer, v1)
        assert_kind_of(Integer, v2)
        assert(v1 == v2)
    end


    def test_context_synchronize
        s = CUContext.synchronize
        assert_nil(s)
    end

end
