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

require 'delegate'
require 'cuda/runtime/ffi-cuda'
require 'memory/buffer'
require 'helpers/struct'


module SGC
module Cuda

    include SGC::Memory

    class CudaError < API::Enum; end # @see API::CudaError
    class CudaDeviceFlags < API::Enum; end # @see API::CudaDeviceFlags
    class CudaEventFlags < API::Enum; end # @see API::CudaEventFlags
    class CudaHostAllocFlags < API::Enum; end # @see API::CudaHostAllocFlags
    class CudaHostRegisterFlags < API::Enum; end # @see API::CudaHostRegisterFlags
    class CudaArrayFlags < API::Enum; end # @see API::CudaArrayFlags
    class CudaMemoryType < API::Enum; end  # @see API::CudaMemoryType
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

    class Dim3 < DelegateClass(API::Dim3); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaDeviceProp < DelegateClass(API::CudaDeviceProp); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaFunctionAttributes < DelegateClass(API::CudaFunctionAttributes); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaChannelFormatDesc < DelegateClass(API::CudaChannelFormatDesc); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaPitchedPtr < DelegateClass(API::CudaPitchedPtr); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaPos < DelegateClass(API::CudaPos); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaExtent < DelegateClass(API::CudaExtent); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class CudaMemcpy3DParms < DelegateClass(API::CudaMemcpy3DParms); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class TextureReference < DelegateClass(API::TextureReference); end # See {file:lib/cuda/runtime/ffi-cuda.rb}
    class SurfaceReference < DelegateClass(API::SurfaceReference); end  # See {file:lib/cuda/runtime/ffi-cuda.rb}

    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(Dim3, API::Dim3)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaDeviceProp, API::CudaDeviceProp)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaFunctionAttributes, API::CudaFunctionAttributes)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaChannelFormatDesc, API::CudaChannelFormatDesc)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaPitchedPtr, API::CudaPitchedPtr)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaPos, API::CudaPos)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaExtent, API::CudaExtent)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(CudaMemcpy3DParms, API::CudaMemcpy3DParms)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(TextureReference, API::TextureReference)
    SGC::Helper::Struct::Pvt::define_delegated_struct_methods(SurfaceReference, API::SurfaceReference)

end # module
end # module
