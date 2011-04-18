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

    CudaError = API::CudaError
    CudaDeviceFlags = API::CudaDeviceFlags
    CudaEventFlags = API::CudaEventFlags
    CudaHostAllocFlags = API::CudaHostAllocFlags
    CudaHostRegisterFlags = API::CudaHostRegisterFlags
    CudaArrayFlags = API::CudaArrayFlags
    CudaMemoryType = API::CudaMemoryType
    CudaMemcpyKind = API::CudaMemcpyKind
    CudaChannelFormatKind = API::CudaChannelFormatKind
    CudaFuncCache = API::CudaFuncCache
    CudaLimit = API::CudaLimit
    CudaOutputMode = API::CudaOutputMode
    CudaComputeMode = API::CudaComputeMode
    CudaSurfaceBoundaryMode = API::CudaSurfaceBoundaryMode
    CudaSurfaceFormatMode = API::CudaSurfaceFormatMode
    CudaTextureAddressMode = API::CudaTextureAddressMode
    CudaTextureFilterMode = API::CudaTextureFilterMode
    CudaTextureReadMode = API::CudaTextureReadMode

    Dim3 = API::Dim3
    CudaDeviceProp = API::CudaDeviceProp
    CudaFuncAttributes = API::CudaFuncAttributes
    CudaChannelFormatDesc = API::CudaChannelFormatDesc
    CudaPitchedPtr = API::CudaPitchedPtr
    CudaPos = API::CudaPos
    CudaExtent = API::CudaExtent
    CudaMemcpy3DParms = API::CudaMemcpy3DParms
    TextureReference = API::TextureReference
    SurfaceReference = API::SurfaceReference

end # module
end # module
