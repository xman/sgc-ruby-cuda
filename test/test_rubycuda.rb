#-----------------------------------------------------------------------
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
#-----------------------------------------------------------------------

require 'test/unit'
require 'tempfile'
require 'rubycuda'
require 'memory/pointer'

include SGC::Cuda


class TestRubyCuda < Test::Unit::TestCase

    def setup
        CudaDevice.current = ENV['DEVID'].to_i
    end

    def teardown
        CudaThread.exit
    end


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


    def test_version
        dv = driver_version
        rv = runtime_version
        assert(dv > 0)
        assert(rv > 0)
    end


    def test_device_count
        count = CudaDevice.count
        assert(count > 0, "Device count failed.")
    end

    def test_device_get_set
        count = CudaDevice.count
        (0...count).each do |devid|
            r = CudaDevice.set(devid)
            assert_equal(CudaDevice, r)
            d = CudaDevice.get
            assert_equal(devid, d)
        end

        count = CudaDevice.count
        (0...count).each do |devid|
            r = CudaDevice.current = devid
            assert_equal(devid, r)
            d = CudaDevice.current
            assert_equal(devid, d)
        end
    end

    def test_device_choose
        count = CudaDevice.count
        prop = CudaDeviceProp.new
        devid = CudaDevice.choose(prop)
        assert(devid >= 0 && devid < count)
    end

    def test_device_properties
        prop = CudaDevice.properties
        assert_instance_of(CudaDeviceProp, prop)
        # TODO: assert the content of the _prop_.
    end

    def test_device_flags
        CudaDeviceFlags.symbols.each do |k|
            r = CudaDevice.flags = k
            assert_equal(k, r)
            r = CudaDevice.flags = CudaDeviceFlags[k]
            assert_equal(CudaDeviceFlags[k], r)
        end
    end

    def test_device_valid_devices
        count = CudaDevice.count
        devs = []
        (0...count).each do |devid|
            devs << devid
        end
        r = CudaDevice.valid_devices = devs
        assert_equal(devs, r)
    end

    def test_thread_exit
        r = CudaThread.exit
        assert_equal(CudaThread, r)
    end

    def test_thread_cache_config
        CudaFuncCache.symbols.each do |k|
            r = CudaThread.cache_config = k
            assert_equal(k, r)
            c = CudaThread.cache_config
            assert_equal(k, c)
        end
    end

    def test_thread_limit
        CudaLimit.symbols.each do |k|
            v = CudaThread.limit(k)
            assert_kind_of(Integer, v)
            r = CudaThread.limit = [k, v]
            assert_equal([k, v], r)
        end
    end

    def test_thread_synchronize
        r = CudaThread.synchronize
        assert_equal(CudaThread, r)
    end

    def test_buffer_initialize
        bint = Buffer.new(:int, 16)
        assert_instance_of(Buffer, bint)
        assert_equal(16, bint.size)
        assert_equal(4, bint.element_size)
        assert_equal(4, Buffer.element_size(:int))

        blong = Buffer.new(:long, 10)
        assert_instance_of(Buffer, blong)
        assert_equal(10, blong.size)
        # TODO: Detect if this is 32bit or 64bit OS and check accordingly.
        assert(blong.element_size == 4 || blong.element_size == 8)
        assert(Buffer.element_size(:long) == 4 || Buffer.element_size(:long) == 8)

        bfloat = Buffer.new(:float, 20)
        assert_instance_of(Buffer, bfloat)
        assert_equal(20, bfloat.size)
        assert_equal(4, bfloat.element_size)
        assert_equal(4, Buffer.element_size(:float))
    end

    def test_buffer_access
        b = Buffer.new(:int, 16)
        b[0] = 10
        assert_equal(10, b[0])
        b[9] = 20
        assert_equal(20, b[9])

        b = Buffer.new(:float, 10)
        b[2] = 3.14
        assert_in_delta(3.14, b[2])
        b[8] = 9.33
        assert_in_delta(9.33, b[8])
    end

    def test_buffer_ptr
        b = Buffer.new(:int, 32)
        p = b.ptr
        assert_not_nil(p)
    end

    def test_buffer_offset
        size = 16
        b = Buffer.new(:int, size)
        (0...size).each do |i|
            b[i] = i
        end

        m0 = b.offset(0)
        m8 = b.offset(8)
        assert_equal(m0.ptr, m8.offset(-8*b.element_size).ptr)
    end

    def test_cuda_device_memory_malloc_free
        p = CudaDeviceMemory.malloc(1024)
        assert_instance_of(SGC::Memory::MemoryPointer, p)
        r = CudaDeviceMemory.free(p)
        assert_nil(r)
    end

    def test_memory_copy
        size = 16
        type = :int
        nbytes = size*Buffer.element_size(type)

        a = Buffer.new(type, size)
        b = Buffer.new(type, size)
        c = Buffer.new(type, size)
        d = Buffer.new(type, size)
        p = CudaDeviceMemory.malloc(nbytes)
        q = CudaDeviceMemory.malloc(nbytes)

        (0...size).each do |i|
            a[i] = 0
            b[i] = 0
            c[i] = 0
            d[i] = 0
        end

        extend CudaMemory

        memcpy_htoh(b, a, nbytes)
        (0...size).each do |i|
            assert_equal(a[i], b[i])
        end

        memcpy_htod(p, b, nbytes)
        memcpy_dtoh(c, p, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], c[i])
        end

        memcpy_dtod(q, p, nbytes)
        CudaThread.synchronize
        memcpy_dtoh(d, q, nbytes)
        (0...size).each do |i|
            assert_equal(b[i], d[i])
        end

        CudaDeviceMemory.free(p)
        CudaDeviceMemory.free(q)
    end

    def test_function_name
        f = CudaFunction.new("vadd")
        assert_equal("vadd", f.name)
    end

    def test_function_attributes
        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)
        f = CudaFunction.new("vadd")
        a = f.attributes
        assert_instance_of(CudaFuncAttributes, a)
        CudaFunction.unload_all_libs
    end

    def test_function_cache_config
        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)
        f = CudaFunction.new("vadd")
        CudaFuncCache.symbols.each do |k|
            f.cache_config = k
        end
        CudaFunction.unload_all_libs
    end

    def test_function_launch
        type = :int
        size = 10
        nbytes = size*4

        a = Buffer.new(type, size)
        b = Buffer.new(type, size)
        c = Buffer.new(type, size)

        p = CudaDeviceMemory.malloc(nbytes)
        q = CudaDeviceMemory.malloc(nbytes)
        r = CudaDeviceMemory.malloc(nbytes)

        (0...size).each do |i|
            a[i] = i
            b[i] = 2
            c[i] = 0
        end

        CudaMemory.memcpy_htod(p, a, nbytes)
        CudaMemory.memcpy_htod(q, b, nbytes)
        CudaMemory.memcpy_htod(r, c, nbytes)

        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)

        CudaFunction.configure(Dim3.new(1, 1, 1), Dim3.new(size, 1, 1))
        CudaFunction.setup(p, q, r, size)

        f = CudaFunction.new("vadd")
        f.launch

        CudaMemory.memcpy_dtoh(c, r, nbytes)

        (0...size).each do |i|
            assert_equal(a[i] + b[i], c[i])
        end

        CudaFunction.unload_all_libs

        CudaDeviceMemory.free(p)
        CudaDeviceMemory.free(q)
        CudaDeviceMemory.free(r)
    end

    def test_stream_create_destroy
        s = CudaStream.new.create
        assert_instance_of(CudaStream, s)
        r = s.destroy
        assert_nil(r)
    end

    def test_stream_query
        s = CudaStream.new.create
        b = s.query
        assert(b)
        s.destroy
    end

    def test_stream_synchronize
        s = CudaStream.new.create
        r = s.synchronize
        assert_instance_of(CudaStream, r)
        s.destroy
    end

    def test_stream_wait_event
        s = CudaStream.new.create
        s.destroy
    end

    def test_event_create_destroy
        e = CudaEvent.new.create
        assert_instance_of(CudaEvent, e)
        r = e.destroy
        assert_nil(r)

        e = CudaEvent.new.create(CudaEventFlags[:cudaEventDefault] | CudaEventFlags[:cudaEventBlockingSync])
        assert_instance_of(CudaEvent, e)
        r = e.destroy
        assert_nil(r)

        e = CudaEvent.new.create(:cudaEventDefault)
        assert_instance_of(CudaEvent, e)
        r = e.destroy
        assert_nil(r)
    end

    def test_event_record_synchronize_query
        e = CudaEvent.new.create
        e = e.record
        assert_instance_of(CudaEvent, e)
        e = e.synchronize
        assert_instance_of(CudaEvent, e)
        b = e.query
        assert(b)
        e.destroy
    end

    def test_event_elapsed_time
        e1 = CudaEvent.new.create
        e2 = CudaEvent.new.create

        e1.record
        e2.record
        e2.synchronize
        t = CudaEvent.elapsed_time(e1, e2)
        assert_kind_of(Numeric, t)

        e1.destroy
        e2.destroy
    end

private

    def prepare_kernel_lib
        src_file = 'test/vadd.cu'
        lib_file = 'test/libvadd.so'
        if File.exists?(lib_file) == false || File.mtime(src_file) > File.mtime(lib_file)
            nvcc_build_dynamic_library(src_file, lib_file)
        end
        lib_file
    end

    def nvcc_build_dynamic_library(src_path, lib_path)
        case RUBY_PLATFORM
            when /darwin/    # Build universal binary for i386 and x86_64 platforms.
                f32 = Tempfile.new("rubycuda_test32.so")
                f64 = Tempfile.new("rubycuda_test64.so")
                f32.close
                f64.close
                system %{nvcc -shared -m32 -Xcompiler -fPIC #{src_path} -o #{f32.path}}
                system %{nvcc -shared -m64 -Xcompiler -fPIC #{src_path} -o #{f64.path}}
                system %{lipo -arch i386 #{f32.path} -arch x86_64 #{f64.path} -create -output #{lib_path}}
            else    # Build default platform binary.
                system %{nvcc -shared -Xcompiler -fPIC #{src_path} -o #{lib_path}}
        end
    end

end
