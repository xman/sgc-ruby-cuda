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
require 'memory/buffer'


module SGC
module Cuda

    include SGC::Memory

    class CudaError < API::Enum; end # @see API::CudaError
    class CudaDeviceFlags < API::Enum; end # @see API::CudaDeviceFlags
    class CudaEventFlags < API::Enum; end # @see API::CudaEventFlags
    class CudaHostAllocFlags < API::Enum; end # @see API::CudaHostAllocFlags
    class CudaHostRegisterFlags < API::Enum; end # @see API::CudaHostRegisterFlags
    class CudaArrayFlags < API::Enum; end # @see API::CudaArrayFlags
    class CudaMemoryType < API::Enum; end # @see API::CudaMemoryType
    class CudaMemcpyKind < API::Enum; end # @see API::CudaMemcpyKind
    class CudaChannelFormatKind < API::Enum; end # @see API::CudaChannelFormatKind
    class CudaFunctionCache < API::Enum; end # @see API::CudaFunctionCache
    class CudaLimit < API::Enum; end # @see API::CudaLimit
    class CudaOutputMode < API::Enum; end # @see API::CudaOutputMode
    class CudaComputeMode < API::Enum; end # @see API::CudaComputeMode
    class CudaSurfaceBoundaryMode < API::Enum; end # @see API::CudaSurfaceBoundaryMode
    class CudaSurfaceFormatMode < API::Enum; end # @see API::CudaSurfaceFormatMode
    class CudaTextureAddressMode < API::Enum; end # @see API::CudaTextureAddressMode
    class CudaTextureFilterMode < API::Enum; end # @see API::CudaTextureFilterMode
    class CudaTextureReadMode < API::Enum; end # @see API::CudaTextureReadMode

    class Dim3 < API::Dim3; end
    class CudaDeviceProp < API::CudaDeviceProp; end
    class CudaFunctionAttributes < API::CudaFunctionAttributes; end
    class CudaChannelFormatDesc < API::CudaChannelFormatDesc; end
    class CudaPitchedPtr < API::CudaPitchedPtr; end
    class CudaPos < API::CudaPos; end
    class CudaExtent < API::CudaExtent; end
    class CudaMemcpy3DParms < API::CudaMemcpy3DParms; end
    class TextureReference < API::TextureReference; end
    class SurfaceReference < API::SurfaceReference; end

end # module
end # module
