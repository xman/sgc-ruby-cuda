#
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
#

require 'ffi'
require 'ffi/prettystruct'


module SGC
module Cuda
module API

    extend FFI::Library
    ffi_lib "cudart"

    CudaError = enum(
        :cudaSuccess, 0,
        :cudaErrorMissingConfiguration, 1,
        :cudaErrorMemoryAllocation, 2,
        :cudaErrorInitializationError, 3,
        :cudaErrorLaunchFailure, 4,
        :cudaErrorPriorLaunchFailure, 5,
        :cudaErrorLaunchTimeout, 6,
        :cudaErrorLaunchOutOfResources, 7,
        :cudaErrorInvalidDeviceFunction, 8,
        :cudaErrorInvalidConfiguration, 9,
        :cudaErrorInvalidDevice, 10,
        :cudaErrorInvalidValue, 11,
        :cudaErrorInvalidPitchValue, 12,
        :cudaErrorInvalidSymbol, 13,
        :cudaErrorMapBufferObjectFailed, 14,
        :cudaErrorUnmapBufferObjectFailed, 15,
        :cudaErrorInvalidHostPointer, 16,
        :cudaErrorInvalidDevicePointer, 17,
        :cudaErrorInvalidTexture, 18,
        :cudaErrorInvalidTextureBinding, 19,
        :cudaErrorInvalidChannelDescriptor, 20,
        :cudaErrorInvalidMemcpyDirection, 21,
        :cudaErrorAddressOfConstant, 22,
        :cudaErrorTextureFetchFailed, 23,
        :cudaErrorTextureNotBound, 24,
        :cudaErrorSynchronizationError, 25,
        :cudaErrorInvalidFilterSetting, 26,
        :cudaErrorInvalidNormSetting, 27,
        :cudaErrorMixedDeviceExecution, 28,
        :cudaErrorCudartUnloading, 29,
        :cudaErrorUnknown, 30,
        :cudaErrorNotYetImplemented, 31,
        :cudaErrorMemoryValueTooLarge, 32,
        :cudaErrorInvalidResourceHandle, 33,
        :cudaErrorNotReady, 34,
        :cudaErrorInsufficientDriver, 35,
        :cudaErrorSetOnActiveProcess, 36,
        :cudaErrorInvalidSurface, 37,
        :cudaErrorNoDevice, 38,
        :cudaErrorECCUncorrectable, 39,
        :cudaErrorSharedObjectSymbolNotFound, 40,
        :cudaErrorSharedObjectInitFailed, 41,
        :cudaErrorUnsupportedLimit, 42,
        :cudaErrorDuplicateVariableName, 43,
        :cudaErrorDuplicateTextureName, 44,
        :cudaErrorDuplicateSurfaceName, 45,
        :cudaErrorDevicesUnavailable, 46,
        :cudaErrorInvalidKernelImage, 47,
        :cudaErrorNoKernelImageForDevice, 48,
        :cudaErrorIncompatibleDriverContext, 49,
        :cudaErrorStartupFailure, 0x7F,
        :cudaErrorApiFailureBase, 10000,
    )
    CudaError_t = CudaError

    CudaDeviceFlags = enum(
        :cudaDeviceScheduleAuto, 0,
        :cudaDeviceScheduleSpin, 1,
        :cudaDeviceScheduleYield, 2,
        :cudaDeviceBlockingSync, 4,
        :cudaDeviceMapHost, 8,
        :cudaDeviceLmemResizeToMax, 16,
    )

    CudaEventFlags = enum(
        :cudaEventDefault, 0,
        :cudaEventBlockingSync, 1,
        :cudaEventDisableTiming, 2,
    )

    CudaHostAllocFlags = enum(
        :cudaHostAllocDefault, 0,
        :cudaHostAllocPortable, 1,
        :cudaHostAllocMapped, 2,
        :cudaHostAllocWriteCombined, 4,
    )

    CudaArrayFlags = enum(
        :cudaArrayDefault, 0x00,
        :cudaArraySurfaceLoadStore, 0x02,
    )

    CudaMemcpyKind = enum(
        :cudaMemcpyHostToHost, 0,
        :cudaMemcpyHostToDevice, 1,
        :cudaMemcpyDeviceToHost, 2,
        :cudaMemcpyDeviceToDevice, 3,
    )

    CudaChannelFormatKind = enum(
        :cudaChannelFormatKindSigned, 0,
        :cudaChannelFormatKindUnsigned, 1,
        :cudaChannelFormatKindFloat, 2,
        :cudaChannelFormatKindNone,3,
    )

    CudaFuncCache = enum(
        :cudaFuncCachePreferNone, 0,
        :cudaFuncCachePreferShared, 1,
        :cudaFuncCachePreferL1, 2,
    )

    CudaLimit = enum(
        :cudaLimitStackSize, 0x00,
        :cudaLimitPrintfFifoSize, 0x01,
        :cudaLimitMallocHeapSize, 0x02,
    )

    CudaComputeMode = enum(
        :cudaComputeModeDefault, 0,
        :cudaComputeModeExclusive, 1,
        :cudaComputeModeProhibited, 2,
    )

    CudaSurfaceBoundaryMode = enum(
        :cudaBoundaryModeZero, 0,
        :cudaBoundaryModeClamp, 1,
        :cudaBoundaryModeTrap, 2,
    )

    CudaSurfaceFormatMode = enum(
        :cudaFormatModeForced, 0,
        :cudaFormatModeAuto, 1,
    )

    CudaTextureAddressMode = enum(
        :cudaAddressModeWrap, 0,
        :cudaAddressModeClamp, 1,
        :cudaAddressModeMirror, 2,
        :cudaAddressModeBorder, 3,
    )

    CudaTextureFilterMode = enum(
        :cudaFilterModePoint, 0,
        :cudaFilterModeLinear, 1,
    )

    CudaTextureReadMode = enum(
        :cudaReadModeElementType, 0,
        :cudaReadModeNormalizedFloat, 1,
    )

    typedef :pointer, :CudaStream
    typedef :pointer, :CudaEvent

    typedef :CudaStream, :CudaStream_t
    typedef :CudaEvent, :CudaEvent_t


    class Dim3 < FFI::Struct
        layout(
            :array, [:uint, 3],
        )

        alias :init :initialize
        alias :get :[]
        alias :set :[]=
        private :init, :get, :set

        def initialize(x, y, z)
            init
            @array = get(:array)
            @array[0], @array[1], @array[2] = x, y, z
        end

        def [](index); @array[index]; end
        def []=(index, value); @array[index] = value; end

        def x; @array[0]; end
        def y; @array[1]; end
        def z; @array[2]; end

        def x=(value); @array[0] = value; end
        def y=(value); @array[1] = value; end
        def z=(value); @array[2] = value; end

    end

    class CudaDeviceProp < FFI::PrettyStruct
        layout(
            :name, [:char, 256],
            :totalGlobalMem, :size_t,
            :sharedMemPerBlock, :size_t,
            :regsPerBlock, :int,
            :warpSize, :int,
            :memPitch, :size_t,
            :maxThreadsPerBlock, :int,
            :maxThreadsDim, [:int, 3],
            :maxGridSize, [:int, 3],
            :clockRate, :int,
            :totalConstMem, :size_t,
            :major, :int,
            :minor, :int,
            :textureAlignment, :size_t,
            :deviceOverlap, :int,
            :multiProcessorCount, :int,
            :kernelExecTimeoutEnabled, :int,
            :integrated, :int,
            :canMapHostMemory, :int,
            :computeMode, :int,
            :maxTexture1D, :int,
            :maxTexture2D, [:int, 2],
            :maxTexture3D, [:int, 3],
            :maxTexture2DArray, [:int, 3],
            :surfaceAlignment, :size_t,
            :concurrentKernels, :int,
            :ECCEnabled, :int,
            :pciBusID, :int,
            :__cudaReserved, [:int, 21],
        )
    end

    class CudaFuncAttributes < FFI::PrettyStruct
        layout(
            :sharedSizeBytes, :size_t,
            :constSizeBytes, :size_t,
            :localSizeBytes, :size_t,
            :maxThreadsPerBlock, :int,
            :numRegs, :int,
            :ptxVersion, :int,
            :binaryVersion, :int,
            :__cudaReserved, [:int, 6],
        )
    end

    class CudaChannelFormatDesc < FFI::PrettyStruct
        layout(
            :x, :int,
            :y, :int,
            :z, :int,
            :w, :int,
            :f, CudaChannelFormatKind,
        )
    end

    class CudaPitchedPtr < FFI::PrettyStruct
        layout(
            :ptr, :pointer,
            :pitch, :size_t,
            :xsize, :size_t,
            :ysize, :size_t,
        )
    end

    class CudaPos < FFI::PrettyStruct
        layout(
            :x, :size_t,
            :y, :size_t,
            :z, :size_t,
        )
    end

    class CudaExtent < FFI::PrettyStruct
        layout(
            :width, :size_t,
            :height, :size_t,
            :depth, :size_t,
        )
    end

    class CudaMemcpy3DParms < FFI::PrettyStruct
        layout(
            :srcArray, :pointer,
            :srcPos, CudaPos,
            :srcPtr, CudaPitchedPtr,
            :dstArray, :pointer,
            :dstPos, CudaPos,
            :dstPtr, CudaPitchedPtr,
            :extent, CudaExtent,
            :kind, CudaMemcpyKind,
        )
    end

    class TextureReference < FFI::PrettyStruct
        layout(
            :normalized, :int,
            :filterMode, CudaTextureFilterMode,
            :addressMode, [CudaTextureAddressMode, 3],
            :channelDesc, CudaChannelFormatDesc,
            :__cudaReserved, [:int, 16],
        )
    end

    class SurfaceReference < FFI::PrettyStruct
        layout(
            :channelDesc, CudaChannelFormatDesc,
        )
    end

    # CUDA Version Management.
    attach_function :cudaDriverGetVersion, [:pointer], :int
    attach_function :cudaRuntimeGetVersion, [:pointer], :int

    # CUDA Error Handling.
    attach_function :cudaGetErrorString, [CudaError], :string
    attach_function :cudaGetLastError, [], :int
    attach_function :cudaPeekAtLastError, [], :int

    # CUDA Device Management.
    attach_function :cudaChooseDevice, [:pointer, :pointer], :int
    attach_function :cudaGetDevice, [:pointer], :int
    attach_function :cudaGetDeviceCount, [:pointer], :int
    attach_function :cudaGetDeviceProperties, [:pointer, :int], :int
    attach_function :cudaSetDevice, [:int], :int
    attach_function :cudaSetDeviceFlags, [:uint], :int
    attach_function :cudaSetValidDevices, [:pointer, :int], :int

    # CUDA Thread Management.
    attach_function :cudaThreadExit, [], :int
    attach_function :cudaThreadGetCacheConfig, [:pointer], :int
    attach_function :cudaThreadGetLimit, [:pointer, CudaLimit], :int
    attach_function :cudaThreadSetCacheConfig, [CudaFuncCache], :int
    attach_function :cudaThreadSetLimit, [CudaLimit, :size_t], :int
    attach_function :cudaThreadSynchronize, [], :int

    # CUDA Memory Management.
    attach_function :cudaFree, [:pointer], :int
    attach_function :cudaFreeArray, [:pointer], :int
    attach_function :cudaFreeHost, [:pointer], :int
    attach_function :cudaGetSymbolAddress, [:pointer, :string], :int
    attach_function :cudaGetSymbolSize, [:pointer, :string], :int
    attach_function :cudaHostAlloc, [:pointer, :size_t, :uint], :int
    attach_function :cudaHostGetDevicePointer, [:pointer, :pointer, :uint], :int
    attach_function :cudaHostGetFlags, [:pointer, :pointer], :int
    attach_function :cudaMalloc, [:pointer, :size_t], :int
    attach_function :cudaMalloc3D, [:pointer, CudaExtent.by_value], :int
    attach_function :cudaMalloc3DArray, [:pointer, :pointer, CudaExtent.by_value, :uint], :int
    attach_function :cudaMallocArray, [:pointer, :pointer, :size_t, :size_t, :uint], :int
    attach_function :cudaMallocHost, [:pointer, :size_t], :int
    attach_function :cudaMallocPitch, [:pointer, :pointer, :size_t, :size_t], :int
    attach_function :cudaMemcpy, [:pointer, :pointer, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpy2D, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpy2DArrayToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpy2DAsync, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpy2DFromArray, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpy2DFromArrayAsync, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpy2DToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpy2DToArrayAsync, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpy3D, [:pointer], :int
    attach_function :cudaMemcpy3DAsync, [:pointer, :CudaStream], :int
    attach_function :cudaMemcpyArrayToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpyAsync, [:pointer, :pointer, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpyFromArray, [:pointer, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpyFromArrayAsync, [:pointer, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpyFromSymbol, [:pointer, :string, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpyFromSymbolAsync, [:pointer, :string, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpyToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpyToArrayAsync, [:pointer, :size_t, :size_t, :pointer, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemcpyToSymbol, [:string, :pointer, :size_t, :size_t, CudaMemcpyKind], :int
    attach_function :cudaMemcpyToSymbolAsync, [:string, :pointer, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :int
    attach_function :cudaMemGetInfo, [:pointer, :pointer], :int
    attach_function :cudaMemset, [:pointer, :int, :size_t], :int
    attach_function :cudaMemset2D, [:pointer, :size_t, :int, :size_t, :size_t], :int
    attach_function :cudaMemset2DAsync, [:pointer, :size_t, :int, :size_t, :size_t, :CudaStream], :int
    attach_function :cudaMemset3D, [CudaPitchedPtr.by_value, :int, CudaExtent.by_value], :int
    attach_function :cudaMemset3DAsync, [CudaPitchedPtr.by_value, :int, CudaExtent.by_value, :CudaStream], :int
    attach_function :cudaMemsetAsync, [:pointer, :int, :size_t, :CudaStream], :int
    # attach_function :make_cudaExtent, [:size_t, :size_t, :size_t], CudaExtent
    # attach_function :make_cudaPitchedPtr, [:pointer, :size_t, :size_t, :size_t], CudaPitchedPtr
    # attach_function :make_cudaPos, [:size_t, :size_t, :size_t], CudaPos

    def make_cudaExtent(w, h, d)
        e = CudaExtent.new
        e[:width], e[:height], e[:depth] = w, h, d
        e
    end

    def make_cudaPitchedPtr(d, p, xsz, ysz)
        s = CudaPitchedPtr.new
        s[:ptr] = d
        s[:pitch] = p
        s[:xsize] = xsz
        s[:ysize] = ysz
        s
    end

    def make_cudaPos(x, y, z)
        p = CudaPos.new
        p[:x] = x
        p[:y] = y
        p[:z] = z
        p
    end

    # CUDA Execution Control.
    attach_function :cudaConfigureCall, [Dim3.by_value, Dim3.by_value, :size_t, :uint], :int
    attach_function :cudaFuncGetAttributes, [:pointer, :string], :int
    attach_function :cudaFuncSetCacheConfig, [:string, CudaFuncCache], :int
    attach_function :cudaLaunch, [:string], :int
    attach_function :cudaSetDoubleForDevice, [:pointer], :int
    attach_function :cudaSetDoubleForHost, [:pointer], :int
    attach_function :cudaSetupArgument, [:pointer, :size_t, :size_t], :int

    # CUDA Stream Management.
    attach_function :cudaStreamCreate, [:pointer], :int
    attach_function :cudaStreamDestroy, [:CudaStream], :int
    attach_function :cudaStreamQuery, [:CudaStream], :int
    attach_function :cudaStreamSynchronize, [:CudaStream], :int
    attach_function :cudaStreamWaitEvent, [:CudaStream, :CudaEvent, :uint], :int

    # CUDA Event Management.
    attach_function :cudaEventCreate, [:pointer], :int
    attach_function :cudaEventCreateWithFlags, [:pointer, :uint], :int
    attach_function :cudaEventDestroy, [:CudaEvent], :int
    attach_function :cudaEventElapsedTime, [:pointer, :CudaEvent, :CudaEvent], :int
    attach_function :cudaEventQuery, [:CudaEvent], :int
    attach_function :cudaEventRecord, [:CudaEvent, :CudaStream], :int
    attach_function :cudaEventSynchronize, [:CudaEvent], :int

    # CUDA Texture Reference Management.
    attach_function :cudaBindTexture, [:pointer, :pointer, :pointer, :pointer, :size_t], :int
    attach_function :cudaBindTexture2D, [:pointer, :pointer, :pointer, :pointer, :size_t, :size_t, :size_t], :int
    attach_function :cudaBindTextureToArray, [:pointer, :pointer, :pointer], :int
    attach_function :cudaCreateChannelDesc, [:int, :int, :int, :int, CudaChannelFormatKind], CudaChannelFormatDesc.by_value
    attach_function :cudaGetChannelDesc, [:pointer, :pointer], :int
    attach_function :cudaGetTextureAlignmentOffset, [:pointer, :pointer], :int
    attach_function :cudaGetTextureReference, [:pointer, :string], :int
    attach_function :cudaUnbindTexture, [:pointer], :int

    # CUDA Surface Reference Management.
    attach_function :cudaBindSurfaceToArray, [:pointer, :pointer, :pointer], :int
    attach_function :cudaGetSurfaceReference, [:pointer, :string], :int

end # module
end # module
end # module
