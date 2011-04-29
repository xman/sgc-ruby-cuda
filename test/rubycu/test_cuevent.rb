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


class TestCUEvent < Test::Unit::TestCase

    include CUTestBase


    def test_event_create_destroy
        e = CUEvent.create
        assert_instance_of(CUEvent, e)
        e = e.destroy
        assert_nil(e)

        e = CUEvent.create(:DEFAULT)
        assert_instance_of(CUEvent, e)
        e = e.destroy
        assert_nil(e)

        e = CUEvent.create(:DEFAULT, :BLOCKING_SYNC)
        assert_instance_of(CUEvent, e)
        e = e.destroy
        assert_nil(e)

        e = CUEvent.create([:BLOCKING_SYNC, :DISABLE_TIMING])
        assert_instance_of(CUEvent, e)
        e = e.destroy
        assert_nil(e)
    end


    def test_event_record_synchronize_query
        e = CUEvent.create(:DEFAULT)
        e = e.record(0)
        assert_instance_of(CUEvent, e)
        e = e.synchronize
        assert_instance_of(CUEvent, e)
        b = e.query
        assert(b)
        e.destroy
    end


    def test_event_elapsed_time
        e1 = CUEvent.create(:DEFAULT)
        e2 = CUEvent.create(:DEFAULT)
        e1.record(0)
        e2.record(0)
        e2.synchronize
        elapsed = CUEvent.elapsed_time(e1, e2)
        assert_instance_of(Float, elapsed)
        e1.destroy
        e2.destroy
    end

end
