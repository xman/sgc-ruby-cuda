require 'test/unit'
require 'rubycu'

include SGC::CU

DEVID = ENV['DEVID'].to_i


class TestRubyCU < Test::Unit::TestCase

    def setup
        @dev = CUDevice.new.get(DEVID)
        @ctx = CUContext.new.create(0, @dev)
        @mod = prepare_loaded_module
    end

    def teardown
        @mod.unload
        @mod = nil
        @ctx.destroy
        @ctx = nil
        @dev = nil
    end

    def test_device_count
        assert_nothing_raised do
            count = CUDevice.get_count
            assert(count > 0, "Device count failed.")
        end
    end

    def test_device_query
        assert_nothing_raised do
            d = CUDevice.new.get(DEVID)
            assert_not_nil(d)
            assert_device_name(d)
            assert_device_memory_size(d)
            assert_device_capability(d)
            CUDeviceAttribute.constants.each do |symbol|
                k = CUDeviceAttribute.const_get(symbol)
                v = d.get_attribute(k)
                assert_instance_of(Fixnum, v)
            end
        end
    end

    def test_context_create_destroy
        assert_nothing_raised do
            c = CUContext.new.create(0, @dev)
            assert_not_nil(c)
            c = c.destroy
            assert_nil(c)
            CUContextFlags.constants.each do |symbol|
                k = CUContextFlags.const_get(symbol)
                c = CUContext.new.create(k, @dev)
                assert_not_nil(c)
                c = c.destroy
                assert_nil(c)
            end
        end
    end

    def test_context_attach_detach
        assert_nothing_raised do
            c1 = @ctx.attach(0)
            assert_not_nil(c1)
            c2 = @ctx.detach
            assert_nil(c2)
        end
    end

    def test_context_attach_nonzero_flags_detach
        assert_raise(CUInvalidValueError) do
            c1 = @ctx.attach(999)
            assert_not_nil(c1)
            c2 = @ctx.detach
            assert_nil(c2)
        end
    end

    def test_context_push_pop_current
        assert_nothing_raised do
            c1 = CUContext.pop_current
            assert_not_nil(c1)
            c2 = @ctx.push_current
            assert_not_nil(c2)
        end
    end

    def test_context_get_device
        assert_nothing_raised do
            d = CUContext.get_device
            assert_device(d)
        end
    end

    def test_context_get_set_limit
        if @dev.compute_capability[:major] >= 2
            assert_limit = Proc.new { |&b| assert_nothing_raised(&b) }
        else
            assert_limit = Proc.new { |&b| assert_raise(CUUnsupportedLimitError, &b) }
        end
        assert_limit.call do
            stack_size = CUContext.get_limit(CULimit::STACK_SIZE)
            assert_kind_of(Numeric, stack_size)
            fifo_size = CUContext.get_limit(CULimit::PRINTF_FIFO_SIZE)
            assert_kind_of(Numeric, fifo_size)
            s1 = CUContext.set_limit(CULimit::STACK_SIZE, stack_size)
            assert_nil(s1)
            s2 = CUContext.set_limit(CULimit::PRINTF_FIFO_SIZE, fifo_size)
            assert_nil(s2)
        end
    end

    def test_context_synchronize
        assert_nothing_raised do
            s = CUContext.synchronize
            assert_nil(s)
        end
    end

    def test_module_load_unload
        assert_nothing_raised do
            path = prepare_ptx
            m = CUModule.new
            m = m.load(path)
            assert_not_nil(m)
            m = m.unload
            assert_not_nil(m)
        end
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
        assert_nothing_raised do
            m = CUModule.new
            File.open(path) do |f|
                str = f.read
                m.load_data(str)
                m.unload
            end
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
        assert_nothing_raised do
            f = @mod.get_function('vadd')
            assert_not_nil(f)
        end
    end

    def test_module_get_function_with_invalid_name
        assert_raise(CUInvalidValueError) do
            f = @mod.get_function('')
        end

        assert_raise(CUReferenceNotFoundError) do
            f = @mod.get_function('badname')
        end
    end

    def test_module_get_global
        assert_nothing_raised do
            devptr, nbytes = @mod.get_global('gvar')
            assert_equal(4, nbytes)
            u = Int32Buffer.new(1)
            memcpy_dtoh(u, devptr, nbytes)
            assert_equal(1997, u[0])
        end
    end

    def test_module_get_global_with_invalid_name
        assert_raise(CUInvalidValueError) do
            devptr, nbytes = @mod.get_global('')
        end

        assert_raise(CUReferenceNotFoundError) do
            devptr, nbytes = @mod.get_global('badname')
        end
    end

    def test_module_get_texref
        assert_nothing_raised do
            tex = @mod.get_texref('tex')
            assert_not_nil(tex)
        end
    end

    def test_module_get_texref_with_invalid_name
        assert_raise(CUInvalidValueError) do
            tex = @mod.get_texref('')
        end

        assert_raise(CUReferenceNotFoundError) do
            tex = @mod.get_texref('badname')
        end
    end

private

    def assert_device(dev)
        assert_device_name(dev)
        assert_device_memory_size(dev)
        assert_device_capability(dev)
    end

    def assert_device_name(dev)
        assert(dev.get_name.size > 0, "Device name failed.")
    end

    def assert_device_memory_size(dev)
        assert(dev.total_mem > 0, "Device total memory size failed.")
    end

    def assert_device_capability(dev)
        cap = dev.compute_capability
        assert(cap[:major] > 0 && cap[:minor] >= 0, "Device compute capability failed.")
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
