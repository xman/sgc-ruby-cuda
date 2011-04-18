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

require 'cuda/driver/ffi-cu'
require 'memory/buffer'


module SGC
module CU

    include SGC::Memory

    # @see API::CUResult
    # @see SGC::Helper::IEnum::ClassMethods
    class CUResult < API::Enum; end

    # @see API::CUComputeMode
    # @see SGC::Helper::IEnum::ClassMethods
    class CUComputeMode < API::Enum; end

    # @see API::CUDeviceAttribute
    # @see SGC::Helper::IEnum::ClassMethods
    class CUDeviceAttribute < API::Enum; end

    # @see API::CUContextFlags
    # @see SGC::Helper::IEnum::ClassMethods
    class CUContextFlags < API::Enum; end

    # @see API::CULimit
    # @see SGC::Helper::IEnum::ClassMethods
    class CULimit < API::Enum; end

    # @see API::CUFunctionAttribute
    # @see SGC::Helper::IEnum::ClassMethods
    class CUFunctionAttribute < API::Enum; end

    # @see API::CUFunctionCache
    # @see SGC::Helper::IEnum::ClassMethods
    class CUFunctionCache < API::Enum; end

    # @see API::CUFunctionCache
    # @see SGC::Helper::IEnum::ClassMethods
    class CUFuncCache < API::Enum; end

    # @see API::CUEventFlags
    # @see SGC::Helper::IEnum::ClassMethods
    class CUEventFlags < API::Enum; end

    # @see API::CUAddressMode
    # @see SGC::Helper::IEnum::ClassMethods
    class CUAddressMode < API::Enum; end

    # @see API::CUFilterMode
    # @see SGC::Helper::IEnum::ClassMethods
    class CUFilterMode < API::Enum; end

    # @see API::CUTexRefFlags
    # @see SGC::Helper::IEnum::ClassMethods
    class CUTexRefFlags < API::Enum; end

    # @see API::CUArrayFormat
    # @see SGC::Helper::IEnum::ClassMethods
    class CUArrayFormat < API::Enum; end

    # @see API::CUMemoryType
    # @see SGC::Helper::IEnum::ClassMethods
    class CUMemoryType < API::Enum; end

    # @see API::CUPointerAttribute
    # @see SGC::Helper::IEnum::ClassMethods
    class CUPointerAttribute < API::Enum; end

    # @see API::CUJitOption
    # @see SGC::Helper::IEnum::ClassMethods
    class CUJitOption < API::Enum; end

    # @see API::CUJitFallBack
    # @see SGC::Helper::IEnum::ClassMethods
    class CUJitFallBack < API::Enum; end

    # @see API::CUJitTarget
    # @see SGC::Helper::IEnum::ClassMethods
    class CUJitTarget < API::Enum; end

    class CUDevProp < API::CUDevProp; end
    class CudaMemcpy2D < API::CudaMemcpy2D; end
    class CudaMemcpy3D < API::CudaMemcpy3D; end
    class CudaMemcpy3DPeer < API::CudaMemcpy3DPeer; end
    class CudaArrayDescriptor < API::CudaArrayDescriptor; end
    class CudaArray3DDescriptor < API::CudaArray3DDescriptor; end

end # module
end # module
