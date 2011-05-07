#
# Copyright (c) 2011 Chung Shin Yee
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

require 'delegate'
require 'cuda/driver/ffi-cu'
require 'memory/buffer'
require 'helpers/struct'


module SGC
module CU

    include SGC::Memory

    class CUResult < API::Enum; end # @see API::CUResult
    class CUComputeMode < API::Enum; end # @see API::CUComputeMode
    class CUDeviceAttribute < API::Enum; end # @see API::CUDeviceAttribute
    class CUContextFlags < API::Enum; end # @see API::CUContextFlags
    class CULimit < API::Enum; end # @see API::CULimit
    class CUFunctionAttribute < API::Enum; end # @see API::CUFunctionAttribute
    class CUFunctionCache < API::Enum; end # @see API::CUFunctionCache
    class CUEventFlags < API::Enum; end # @see API::CUEventFlags
    class CUAddressMode < API::Enum; end # @see API::CUAddressMode
    class CUFilterMode < API::Enum; end # @see API::CUFilterMode
    class CUTexRefFlags < API::Enum; end # @see API::CUTexRefFlags
    class CUArrayFormat < API::Enum; end # @see API::CUArrayFormat
    class CUMemoryType < API::Enum; end # @see API::CUMemoryType
    class CUPointerAttribute < API::Enum; end # @see API::CUPointerAttribute
    class CUJitOption < API::Enum; end # @see API::CUJitOption
    class CUJitFallBack < API::Enum; end # @see API::CUJitFallBack
    class CUJitTarget < API::Enum; end # @see API::CUJitTarget

    class CUDevProp < DelegateClass(API::CUDevProp); end # See {file:lib/cuda/driver/ffi-cu.rb}
    class CudaMemcpy2D < DelegateClass(API::CudaMemcpy2D); end # See {file:lib/cuda/driver/ffi-cu.rb}
    class CudaMemcpy3D < DelegateClass(API::CudaMemcpy3D); end  # See {file:lib/cuda/driver/ffi-cu.rb}
    class CudaMemcpy3DPeer < DelegateClass(API::CudaMemcpy3DPeer); end # See {file:lib/cuda/driver/ffi-cu.rb}
    class CudaArrayDescriptor < DelegateClass(API::CudaArrayDescriptor); end # See {file:lib/cuda/driver/ffi-cu.rb}
    class CudaArray3DDescriptor < DelegateClass(API::CudaArray3DDescriptor); end # See {file:lib/cuda/driver/ffi-cu.rb}

    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CUDevProp, API::CUDevProp)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaMemcpy2D, API::CudaMemcpy2D)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaMemcpy3D, API::CudaMemcpy3D)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaMemcpy3DPeer, API::CudaMemcpy3DPeer)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaArrayDescriptor, API::CudaArrayDescriptor)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaArray3DDescriptor, API::CudaArray3DDescriptor)

end # module
end # module
