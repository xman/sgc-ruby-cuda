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


module SGC
module CU


class CUDevice

    # See CUDevice::get_count.
    def self.count
        self.get_count
    end

    # See CUDevice#get_name.
    def name
        get_name
    end

    # See CUDevice#get_attribute.
    def attribute(attr)
        get_attribute(attr)
    end

    # See CUDevice#get_properties.
    def properties
        get_properties
    end

end


class CUContext

    # See CUContext::get_device.
    def self.device
        self.get_device
    end

    # See CUContext::get_limit.
    def self.limit(lim)
        get_limit(lim)
    end

    # See CUContext::get_cache_config.
    def self.cache_config
        get_cache_config
    end

    # See CUContext#get_api_version.
    def api_version
        get_api_version
    end

end


class CUModule

    # See CUModule#get_function.
    def function(name_str)
        get_function(name_str)
    end

    # See CUModule#get_global.
    def global(name_str)
        get_global(name_str)
    end

    # See CUModule#get_texref.
    def texref(name_str)
        get_texref(name_str)
    end

end


class CUFunction

    # See CUFunction#get_attribute.
    def attribute(attr)
        get_attribute(attr)
    end

end


class CUTexRef

    # See CUTexRef#get_address.
    def address
        get_address
    end

    # See CUTexRef#get_address_mode.
    def address_mode(dim)
        get_address_mode(dim)
    end

    # See CUTexRef#get_filter_mode.
    def filter_mode
        get_filter_mode
    end

    # See CUTexRef#get_flags.
    def flags
        get_flags
    end

end


# See ::driver_get_version.
def driver_version
    driver_get_version
end
module_function :driver_version


end # module
end # module
