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

end
