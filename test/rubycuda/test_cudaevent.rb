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


class TestCudaEvent < Test::Unit::TestCase

    include CudaTestBase


    def test_event_create_destroy
        e = CudaEvent.create
        assert_instance_of(CudaEvent, e)
        r = e.destroy
        assert_nil(r)

        e = CudaEvent.create(CudaEventFlags.value(:DEFAULT, :BLOCKING_SYNC))
        assert_instance_of(CudaEvent, e)
        r = e.destroy
        assert_nil(r)

        e = CudaEvent.create(:DEFAULT, :BLOCKING_SYNC)
        assert_instance_of(CudaEvent, e)
        r = e.destroy
        assert_nil(r)
    end


    def test_event_record_synchronize_query
        e = CudaEvent.create
        e = e.record
        assert_instance_of(CudaEvent, e)
        e = e.synchronize
        assert_instance_of(CudaEvent, e)
        b = e.query
        assert(b)
        e.destroy
    end


    def test_event_elapsed_time
        e1 = CudaEvent.create
        e2 = CudaEvent.create

        e1.record
        e2.record
        e2.synchronize
        t = CudaEvent.elapsed_time(e1, e2)
        assert_kind_of(Numeric, t)

        e1.destroy
        e2.destroy
    end

end
