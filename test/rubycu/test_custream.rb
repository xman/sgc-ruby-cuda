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


class TestCUStream < Test::Unit::TestCase

    include CUTestBase


    def test_stream_create_destroy
        s = CUStream.create
        assert_instance_of(CUStream, s)
        s = s.destroy
        assert_nil(s)

        s = CUStream.create(0)
        assert_instance_of(CUStream, s)
        s = s.destroy
        assert_nil(s)
    end


    def test_stream_query
        s = CUStream.create
        b = s.query
        assert(b)
        s.destroy
    end


    def test_stream_synchronize
        s = CUStream.create
        s = s.synchronize
        assert_instance_of(CUStream, s)
        s.destroy
    end


    def test_stream_wait_event
        s = CUStream.create
        e = CUEvent.create
        s = s.wait_event(e)
        assert_instance_of(CUStream, s)
        s = s.wait_event(e, 0)
        assert_instance_of(CUStream, s)
        s.destroy
        s = CUStream.wait_event(e)
        assert_nil(s)
        s = CUStream.wait_event(e, 0)
        assert_nil(s)
    end

end
