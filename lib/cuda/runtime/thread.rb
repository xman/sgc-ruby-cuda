#
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
#

require 'cuda/runtime/ffi-cuda'
require 'cuda/runtime/cuda'
require 'cuda/runtime/error'


module SGC
module Cuda

class CudaThread

    def self.exit
        status = API::cudaThreadExit
        Pvt::handle_error(status)
        self
    end


    def self.cache_config
        p = FFI::MemoryPointer.new(:int)
        status = API::cudaThreadGetCacheConfig(p)
        Pvt::handle_error(status)
        CudaFuncCache[p.read_int]
    end


    def self.cache_config=(config)
        status = API::cudaThreadSetCacheConfig(config)
        Pvt::handle_error(status)
        config
    end


    def self.limit(limit)
        p = FFI::MemoryPointer.new(:size_t)
        status = API::cudaThreadGetLimit(p, limit)
        Pvt::handle_error(status)
        p.read_long
    end


    def self.limit=(*limit_value_pair)
        limit, value = limit_value_pair.flatten
        status = API::cudaThreadSetLimit(limit, value)
        Pvt::handle_error(status)
        limit_value_pair
    end


    def self.synchronize
        status = API::cudaThreadSynchronize
        Pvt::handle_error(status)
        self
    end

end

end # module
end # module
