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

require 'rubycu'

include SGC::CU
include SGC::CU::Error

CUInit.init


module CUTestBase

    def setup
        @dev = CUDevice.get(ENV['DEVID'].to_i)
        @ctx = CUContext.create(@dev)
        @mod = prepare_loaded_module
        @func = @mod.function("vadd")
    end

    def teardown
        @func = nil
        @mod.unload
        @mod = nil
        @ctx.destroy
        @ctx = nil
        @dev = nil
    end


    def prepare_ptx
        if File.exists?('test/vadd.ptx') == false || File.mtime('test/vadd.cu') > File.mtime('test/vadd.ptx')
            system %{cd test; nvcc --ptx vadd.cu}
        end
        'test/vadd.ptx'
    end


    def prepare_loaded_module
        path = prepare_ptx
        m = CUModule.new
        m.load(path)
    end


    def assert_device(dev)
        assert_device_name(dev)
        assert_device_memory_size(dev)
        assert_device_capability(dev)
        assert_device_properties(dev)
    end

    def assert_device_name(dev)
        assert(dev.name.size > 0, "Device name failed.")
    end

    def assert_device_memory_size(dev)
        assert(dev.total_mem > 0, "Device total memory size failed.")
    end

    def assert_device_capability(dev)
        cap = dev.compute_capability
        assert(cap[:major] > 0 && cap[:minor] >= 0, "Device compute capability failed.")
    end

    def assert_device_properties(dev)
        p = dev.properties
        assert(p[:clock_rate] > 0)
        assert(p[:max_threads_per_block] > 0)
        assert(p[:mem_pitch] > 0)
        assert(p[:regs_per_block] > 0)
        assert(p[:shared_mem_per_block] > 0)
        assert(p[:simd_width] > 0)
        assert(p[:texture_align] > 0)
        assert(p[:total_constant_memory] > 0)
        assert_equal(3, p[:max_grid_size].count)
        assert_kind_of(Integer, p[:max_grid_size][0])
        assert_kind_of(Integer, p[:max_grid_size][1])
        assert_kind_of(Integer, p[:max_grid_size][2])
        assert_equal(3, p[:max_threads_dim].count)
        assert_kind_of(Integer, p[:max_threads_dim][0])
        assert_kind_of(Integer, p[:max_threads_dim][1])
        assert_kind_of(Integer, p[:max_threads_dim][2])
    end

end
