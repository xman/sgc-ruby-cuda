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
require 'rubycu'
require 'memory/pointer'

include SGC::CU
include SGC::CU::Error

CUInit.init


class TestRubyCU < Test::Unit::TestCase

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

    def test_version
        v = driver_version
        assert_kind_of(Integer, v)
    end

    def test_device_count
        count = CUDevice.count
        assert(count > 0, "Device count failed.")
    end

    def test_device_query
        d = @dev
        assert_instance_of(CUDevice, d)
        assert_device_name(d)
        assert_device_memory_size(d)
        assert_device_capability(d)
        assert_device_properties(d)
        CUDeviceAttribute.symbols.each do |k|
            v = d.attribute(k)
            assert_instance_of(Fixnum, v)
        end
    end

    def test_device_malloc_free
        p = CUDevice.malloc(1024)
        assert_instance_of(CUDevicePtr, p)
        r = p.free
        assert_nil(r)
    end

    def test_device_malloc_with_huge_mem
        assert_raise(CUOutOfMemoryError) do
            size = @dev.total_mem + 1
            CUDevice.malloc(size)
        end
    end

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

    def test_module_load_unload
        path = prepare_ptx
        m = CUModule.new
        m = m.load(path)
        assert_instance_of(CUModule, m)
        m = m.unload
        assert_instance_of(CUModule, m)
    end

    def test_module_load_with_invalid_ptx
        assert_raise(CUInvalidImageError) do
            m = CUModule.new
            m = m.load('test/bad.ptx')
        end

        assert_raise(CUFileNotFoundError) do
            m = CUModule.new
            m = m.load('test/notexists.ptx')
        end
    end

    def test_module_load_data
        path = prepare_ptx
        m = CUModule.new
        File.open(path) do |f|
            str = f.read
            r = m.load_data(str)
            assert_instance_of(CUModule, r)
            m.unload
        end
    end

    def test_module_load_data_with_invalid_ptx
        assert_raise(CUInvalidImageError) do
            m = CUModule.new
            str = "invalid ptx"
            m.load_data(str)
        end
    end

    def test_module_get_function
        f = @mod.function('vadd')
        assert_instance_of(CUFunction, f)
    end

    def test_module_get_function_with_invalid_name
        assert_raise(CUInvalidValueError) do
            f = @mod.function('')
        end

        assert_raise(CUReferenceNotFoundError) do
            f = @mod.function('badname')
        end
    end

    def test_module_get_global
        devptr, nbytes = @mod.global('gvar')
        assert_equal(4, nbytes)
        u = Buffer.new(:int, 1)
        CUMemory.memcpy_dtoh(u, devptr, nbytes)
        assert_equal(1997, u[0])
    end

    def test_module_get_global_with_invalid_name
        assert_raise(CUInvalidValueError) do
            devptr, nbytes = @mod.global('')
        end

        assert_raise(CUReferenceNotFoundError) do
            devptr, nbytes = @mod.global('badname')
        end
    end

#    def test_module_get_texref
#        assert_nothing_raised do
#            tex = @mod.texref('tex')
#            assert_not_nil(tex)
#        end
#    end

#    def test_module_get_texref_with_invalid_name
#        assert_raise(CUInvalidValueError) do
#            tex = @mod.texref('')
#        end

#        assert_raise(CUReferenceNotFoundError) do
#            tex = @mod.texref('badname')
#        end
#    end

    def test_device_ptr_offset
        devptr = CUDevice.malloc(1024)
        p = devptr.offset(1024)
        assert_instance_of(CUDevicePtr, p)
        p = devptr.offset(-1024)
        assert_instance_of(CUDevicePtr, p)
        devptr.free
    end

    def test_function_set_param
        da = CUDevice.malloc(1024)
        db = CUDevice.malloc(1024)
        dc = CUDevice.malloc(1024)
        @func.param = []
        @func.param = [da, db, dc, 10]

        f = @func.param_setf(0, 2.5)
        assert_instance_of(CUFunction, f)
        f = @func.param_seti(4, 33)
        assert_instance_of(CUFunction, f)
        v = Buffer.new(:int, 1)
        f = @func.param_setv(0, v, 4)
        assert_instance_of(CUFunction, f)
        f = @func.param_set_size(8)
        assert_instance_of(CUFunction, f)

        da.free
        db.free
        dc.free
    end

    def test_function_set_block_shape
        @func.block_shape = 2
        @func.block_shape = [2, 3]
        @func.block_shape = 2, 3, 4
    end

    def test_function_set_shared_size
        @func.shared_size = 0
        @func.shared_size = 1024
    end

    def test_function_launch
        assert_function_launch(10) do |f, params|
            f.param = params
            f.block_shape = 10
            f.launch
        end
    end

    def test_function_launch_grid
        assert_function_launch(10) do |f, params|
            f.param = params
            f.block_shape = 10
            f.launch_grid(1)
        end

        assert_function_launch(20) do |f, params|
            f.param = params
            f.block_shape = 5
            f.launch_grid(4)
        end
    end

    def test_function_launch_kernel
        assert_function_launch(30) do |f, params|
            f.launch_kernel(3, 1, 1, 10, 1, 1, 0, 0, params)
        end
    end

    def test_function_launch_async
        assert_nothing_raised do
            assert_function_launch(10) do |f, params|
                f.param = params
                f.block_shape = 10
                f.launch_grid_async(1, 0)
            end

            assert_function_launch(20) do |f, params|
                f.param = params
                f.block_shape = 5
                f.launch_grid_async(4, 0)
            end
        end
    end

    def test_function_get_attribute
        CUFunctionAttribute.symbols.each do |k|
            v = @func.attribute(k)
            assert_instance_of(Fixnum, v)
        end
    end

    def test_function_set_cache_config
        CUFunctionCache.symbols.each do |k|
            @func.cache_config = k
        end
    end

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

=begin
    def test_texref_get_set_address
        assert_nothing_raised do
            t = @mod.get_texref("tex")
            p = CUDevicePtr.new.mem_alloc(16)
            t.set_address(p, 16)
            r = t.get_address
            assert_instance_of(CUDevicePtr, r)
            p.mem_free
        end
    end

    def test_texref_get_set_address_mode
        assert_nothing_raised do
            t = @mod.get_texref("tex")
            r = t.get_address_mode(0)
            assert_const_in_class(CUAddressMode, r)
            t = t.set_address_mode(0, r)
            assert_instance_of(CUTexRef, t)
        end
    end

    def test_texref_get_set_filter_mode
        assert_nothing_raised do
            t = @mod.get_texref("tex")
            r = t.get_filter_mode
            assert_const_in_class(CUFilterMode, r)
            t = t.set_filter_mode(r)
            assert_instance_of(CUTexRef, t)
        end
    end

    def test_texref_get_set_flags
        assert_nothing_raised do
            t = @mod.get_texref("tex")
            f = t.get_flags
            assert_kind_of(Numeric, f)
            t = t.set_flags(f)
            assert_instance_of(CUTexRef, t)
        end
    end
=end

    def test_buffer_initialize
        b = Buffer.new(:int, 10)
        assert_instance_of(Buffer, b)
        # assert_equal(false, b.page_locked?)
        assert_equal(10, b.size)

        b = Buffer.new(:long, 20)
        assert_instance_of(Buffer, b)
        # assert_equal(false, b.page_locked?)
        assert_equal(20, b.size)

        b = Buffer.new(:float, 30)
        assert_instance_of(Buffer, b)
        # assert_equal(false, b.page_locked?)
        assert_equal(30, b.size)
    end

    def test_buffer_initialize_page_locked
    end

    def test_buffer_element_size
        assert_equal(4, Buffer.element_size(:int))
        assert_equal(4, Buffer.element_size(:float))
    end

    def test_buffer_offset
        b = Buffer.new(:int, 16)
        c = b.offset(4)
        assert_kind_of(MemoryPointer, c)

        b = Buffer.new(:int, 10)
        c = b.offset(5)
        assert_kind_of(MemoryPointer, c)

        b = Buffer.new(:long, 10)
        c = b.offset(3)
        assert_kind_of(MemoryPointer, c)

        b = Buffer.new(:float, 10)
        c = b.offset(4)
        assert_kind_of(MemoryPointer, c)
    end

    def test_buffer_access
        b = Buffer.new(:int, 10)
        b[0] = 10
        assert_equal(10, b[0])
        b[9] = 20
        assert_equal(20, b[9])

        b = Buffer.new(:long, 10)
        b[3] = 2**40
        assert_equal(2**40, b[3])
        b[7] = 2**50
        assert_equal(2**50, b[7])

        b = Buffer.new(:float, 10)
        b[2] = 3.14
        assert_in_delta(3.14, b[2])
        b[8] = 9.33
        assert_in_delta(9.33, b[8])
    end

    def test_memory_copy
        type = :int
        size = 16
        element_size = 4
        nbytes = size*element_size
        # b = Buffer.new(type, size, page_locked: true)
        # c = Buffer.new(type, size, page_locked: true)
        b = Buffer.new(type, size)
        c = Buffer.new(type, size)
        d = Buffer.new(type, size)
        e = Buffer.new(type, size)
        p = CUDevice.malloc(nbytes)
        q = CUDevice.malloc(nbytes)
        r = CUDevice.malloc(nbytes)

        (0...size).each do |i|
            b[i] = i
            c[i] = 0
            d[i] = 0
            e[i] = 0
        end

        CUMemory.memcpy_htod(p, b, nbytes)
        CUMemory.memcpy_dtoh(c, p, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], c[i])
        end

        (0...size).each do |i|
            b[i] = 2*i
            c[i] = 0
        end
        CUMemory.memcpy_htod_async(p, b, nbytes, 0)
        CUMemory.memcpy_dtoh_async(c, p, nbytes, 0)
        CUContext.synchronize
        (0...size).each do |i|
            assert_equal(b[i], c[i])
        end

        CUMemory.memcpy_dtod(q, p, nbytes)
        CUContext.synchronize
        CUMemory.memcpy_dtoh(d, q, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], d[i])
        end

        CUMemory.memcpy_dtod_async(r, p, size*element_size, 0)
        CUContext.synchronize
        CUMemory.memcpy_dtoh(e, r, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], e[i])
        end

        if false    # FIXME: The memcpy is not working.
        if @dev.attribute(:UNIFIED_ADDRESSING) > 0
            (0...size).each do |i|
                b[i] = i
                c[i] = 0
                d[i] = 0
                e[i] = 0
            end
            CUMemory.memcpy(p, b, nbytes)
            CUMemory.memcpy(c, p, nbytes)
            (0...size).each do |i|
                assert_equal(b[i], c[i])
            end
        end
        end

        p.free
        q.free
        r.free
    end

    def test_memory_get_info
        info = CUMemory.mem_info
        assert(info[:free] >= 0)
        assert(info[:total] >= 0)
    end

private

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

    def assert_function_launch(size)
        type = :int
        element_size = 4
        nbytes = size*element_size
        da = CUDevice.malloc(nbytes)
        db = CUDevice.malloc(nbytes)
        dc = CUDevice.malloc(nbytes)
        ha = Buffer.new(type, size)
        hb = Buffer.new(type, size)
        hc = Array.new(size)
        hd = Buffer.new(type, size)
        (0...size).each { |i|
            ha[i] = i
            hb[i] = 1
            hc[i] = ha[i] + hb[i]
            hd[i] = 0
        }
        CUMemory.memcpy_htod(da, ha, nbytes)
        CUMemory.memcpy_htod(db, hb, nbytes)

        yield @func, [da, db, dc, size]

        CUMemory.memcpy_dtoh(hd, dc, nbytes)
        (0...size).each { |i|
            assert_equal(hc[i], hd[i])
        }
        da.free
        db.free
        dc.free
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

end
