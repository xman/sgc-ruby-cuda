require 'test/unit'
require 'rubycu'

include SGC::CU

DEVID = ENV['DEVID'].to_i


class TestRubyCU < Test::Unit::TestCase

    def setup
        @dev = CUDevice.new.get(DEVID)
        @ctx = CUContext.new.create(0, @dev)
    end

    def teardown
        @ctx.destroy
        @ctx = nil
        @dev = nil
    end

    def test_device_count
        assert_nothing_raised do
            count = CUDevice.get_count
            assert(count > 0)
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

private

    def assert_device(dev)
        assert_device_name(dev)
        assert_device_memory_size(dev)
        assert_device_capability(dev)
    end

    def assert_device_name(dev)
        assert(dev.get_name.size > 0)
    end

    def assert_device_memory_size(dev)
        assert(dev.total_mem > 0)
    end

    def assert_device_capability(dev)
        cap = dev.compute_capability
        assert(cap[:major] > 0 && cap[:minor] > 0)
    end

end
