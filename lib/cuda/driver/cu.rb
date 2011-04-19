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
    class CUResult < API::Enum; end

    # @see API::CUComputeMode
    class CUComputeMode < API::Enum; end

    # @see API::CUDeviceAttribute
    class CUDeviceAttribute < API::Enum; end

    # @see API::CUContextFlags
    class CUContextFlags < API::Enum; end

    # @see API::CULimit
    class CULimit < API::Enum; end

    # @see API::CUFunctionAttribute
    class CUFunctionAttribute < API::Enum; end

    # @see API::CUFunctionCache
    class CUFunctionCache < API::Enum; end

    # @see API::CUEventFlags
    class CUEventFlags < API::Enum; end

    # @see API::CUAddressMode
    class CUAddressMode < API::Enum; end

    # @see API::CUFilterMode
    class CUFilterMode < API::Enum; end

    # @see API::CUTexRefFlags
    class CUTexRefFlags < API::Enum; end

    # @see API::CUArrayFormat
    class CUArrayFormat < API::Enum; end

    # @see API::CUMemoryType
    class CUMemoryType < API::Enum; end

    # @see API::CUPointerAttribute
    class CUPointerAttribute < API::Enum; end

    # @see API::CUJitOption
    class CUJitOption < API::Enum; end

    # @see API::CUJitFallBack
    class CUJitFallBack < API::Enum; end

    # @see API::CUJitTarget
    class CUJitTarget < API::Enum; end

    class CUDevProp < API::CUDevProp; end
    class CudaMemcpy2D < API::CudaMemcpy2D; end
    class CudaMemcpy3D < API::CudaMemcpy3D; end
    class CudaMemcpy3DPeer < API::CudaMemcpy3DPeer; end
    class CudaArrayDescriptor < API::CudaArrayDescriptor; end
    class CudaArray3DDescriptor < API::CudaArray3DDescriptor; end

end # module
end # module
