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

require 'ffi'
require 'ffi/prettystruct'
require 'helpers/interface/ienum'
require 'helpers/flags'
require 'helpers/klass'


module SGC
module CU
module API

    extend FFI::Library
    ffi_lib "cuda"

    class Enum
        extend SGC::Helper::IEnum
        extend SGC::Helper::FlagsValue

        def self.inherited(subclass)
            subclass.instance_eval %{
                def symbols
                    SGC::CU::API::#{SGC::Helper.classname(subclass)}.symbols
                end

                def [](*args)
                    SGC::CU::API::#{SGC::Helper.classname(subclass)}[*args]
                end
            }
        end
    end

    CUResult = enum(
        :SUCCESS, 0,
        :ERROR_INVALID_VALUE, 1,
        :ERROR_OUT_OF_MEMORY, 2,
        :ERROR_NOT_INITIALIZED, 3,
        :ERROR_DEINITIALIZED, 4,
        :ERROR_PROFILER_DISABLED, 5,
        :ERROR_PROFILER_NOT_INITIALIZED, 6,
        :ERROR_PROFILER_ALREADY_STARTED, 7,
        :ERROR_PROFILER_ALREADY_STOPPED, 8,
        :ERROR_NO_DEVICE, 100,
        :ERROR_INVALID_DEVICE, 101,
        :ERROR_INVALID_IMAGE, 200,
        :ERROR_INVALID_CONTEXT, 201,
        :ERROR_CONTEXT_ALREADY_CURRENT, 202, # Deprecated.
        :ERROR_MAP_FAILED, 205,
        :ERROR_UNMAP_FAILED, 206,
        :ERROR_ARRAY_IS_MAPPED, 207,
        :ERROR_ALREADY_MAPPED, 208,
        :ERROR_NO_BINARY_FOR_GPU, 209,
        :ERROR_ALREADY_ACQUIRED, 210,
        :ERROR_NOT_MAPPED, 211,
        :ERROR_NOT_MAPPED_AS_ARRAY, 212,
        :ERROR_NOT_MAPPED_AS_POINTER, 213,
        :ERROR_ECC_UNCORRECTABLE, 214,
        :ERROR_UNSUPPORTED_LIMIT, 215,
        :ERROR_CONTEXT_ALREADY_IN_USE, 216,
        :ERROR_INVALID_SOURCE, 300,
        :ERROR_FILE_NOT_FOUND, 301,
        :ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, 302,
        :ERROR_SHARED_OBJECT_INIT_FAILED, 303,
        :ERROR_OPERATING_SYSTEM, 304,
        :ERROR_INVALID_HANDLE, 400,
        :ERROR_NOT_FOUND, 500,
        :ERROR_NOT_READY, 600,
        :ERROR_LAUNCH_FAILED, 700,
        :ERROR_LAUNCH_OUT_OF_RESOURCES, 701,
        :ERROR_LAUNCH_TIMEOUT, 702,
        :ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, 703,
        :ERROR_PEER_ACCESS_ALREADY_ENABLED, 704,
        :ERROR_PEER_ACCESS_NOT_ENABLED, 705,
        :ERROR_PRIMARY_CONTEXT_ACTIVE, 708,
        :ERROR_CONTEXT_IS_DESTROYED, 709,
        :ERROR_UNKNOWN, 999,
    )

    CUComputeMode = enum(
        :DEFAULT, 0,
        :EXCLUSIVE, 1,
        :PROHIBITED, 2,
        :EXCLUSIVE_PROCESS, 3,
    )

    CUDeviceAttribute = enum(
        :MAX_THREADS_PER_BLOCK, 1,
        :MAX_BLOCK_DIM_X, 2,
        :MAX_BLOCK_DIM_Y, 3,
        :MAX_BLOCK_DIM_Z, 4,
        :MAX_GRID_DIM_X, 5,
        :MAX_GRID_DIM_Y, 6,
        :MAX_GRID_DIM_Z, 7,
        :MAX_SHARED_MEMORY_PER_BLOCK, 8,
        :SHARED_MEMORY_PER_BLOCK, 8, # Deprecated. Use :MAX_SHARED_MEMORY_PER_BLOCK.
        :TOTAL_CONSTANT_MEMORY, 9,
        :WARP_SIZE, 10,
        :MAX_PITCH, 11,
        :MAX_REGISTERS_PER_BLOCK, 12,
        :REGISTERS_PER_BLOCK, 12, # Deprecated. Use :MAX_REGISTERS_PER_BLOCK.
        :CLOCK_RATE, 13,
        :TEXTURE_ALIGNMENT, 14,
        :GPU_OVERLAP, 15, # Deprecated. Use :ASYNC_ENGINE_COUNT.
        :MULTIPROCESSOR_COUNT, 16,
        :KERNEL_EXEC_TIMEOUT, 17,
        :INTEGRATED, 18,
        :CAN_MAP_HOST_MEMORY, 19,
        :COMPUTE_MODE, 20,
        :MAXIMUM_TEXTURE1D_WIDTH, 21,
        :MAXIMUM_TEXTURE2D_WIDTH, 22,
        :MAXIMUM_TEXTURE2D_HEIGHT, 23,
        :MAXIMUM_TEXTURE3D_WIDTH, 24,
        :MAXIMUM_TEXTURE3D_HEIGHT, 25,
        :MAXIMUM_TEXTURE3D_DEPTH, 26,
        :MAXIMUM_TEXTURE2D_LAYERED_WIDTH, 27,
        :MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, 28,
        :MAXIMUM_TEXTURE2D_LAYERED_LAYERS, 29,
        :MAXIMUM_TEXTURE2D_ARRAY_WIDTH, 27, # Deprecated. Use :MAXIMUM_TEXTURE2D_LAYERED_WIDTH.
        :MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, 28, # Deprecated. Use :MAXINUM_TEXTURE2D_LAYERED_HEIGHT.
        :MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, 29, # Deprecated. Use :MAXIMUM_TEXTURE2D_LAYERED_LAYERS.
        :SURFACE_ALIGNMENT, 30,
        :CONCURRENT_KERNELS, 31,
        :ECC_ENABLED, 32,
        :PCI_BUS_ID, 33,
        :PCI_DEVICE_ID, 34,
        :TCC_DRIVER, 35,
        :MEMORY_CLOCK_RATE, 36,
        :GLOBAL_MEMORY_BUS_WIDTH, 37,
        :L2_CACHE_SIZE, 38,
        :MAX_THREADS_PER_MULTIPROCESSOR, 39,
        :ASYNC_ENGINE_COUNT, 40,
        :UNIFIED_ADDRESSING, 41,
        :MAXIMUM_TEXTURE1D_LAYERED_WIDTH, 42,
        :MAXINUM_TEXTURE1D_LAYERED_LAYERS, 43,
    )

    CUContextFlags = enum(
        :SCHED_AUTO, 0x00,
        :SCHED_SPIN, 0x01,
        :SCHED_YIELD, 0x02,
        :SCHED_BLOCKING_SYNC, 0x04,
        :BLOCKING_SYNC, 0x04, # Deprecated. Use :SCHED_BLOCKING_SYNC.
        :MAP_HOST, 0x08,
        :LMEM_RESIZE_TO_MAX, 0x10,
    )

    CULimit = enum(
        :STACK_SIZE, 0x00,
        :PRINTF_FIFO_SIZE, 0x01,
        :MALLOC_HEAP_SIZE, 0x02,
    )

    CUFunctionAttribute = enum(
        :MAX_THREADS_PER_BLOCK, 0,
        :SHARED_SIZE_BYTES, 1,
        :CONST_SIZE_BYTES, 2,
        :LOCAL_SIZE_BYTES, 3,
        :NUM_REGS, 4,
        :PTX_VERSION, 5,
        :BINARY_VERSION, 6,
    )

    CUFunctionCache = enum(
        :PREFER_NONE, 0x00,
        :PREFER_SHARED, 0x01,
        :PREFER_L1, 0x02,
    )

    CUEventFlags = enum(
        :DEFAULT, 0,
        :BLOCKING_SYNC, 1,
        :DISABLE_TIMING, 2,
    )

    CUAddressMode = enum(
        :WRAP, 0,
        :CLAMP, 1,
        :MIRROR, 2,
        :BORDER, 3,
    )

    CUFilterMode = enum(
        :POINT, 0,
        :LINEAR, 1,
    )

    CUTexRefFlags = enum(
        :READ_AS_INTEGER, 0x01,
        :NORMALIZED_COORDINATES, 0x02,
        :SRGB, 0x10,
    )

    CUArrayFormat = enum(
        :UNSIGNED_INT8, 0x01,
        :UNSIGNED_INT16, 0x02,
        :UNSIGNED_INT32, 0x03,
        :SIGNED_INT8, 0x08,
        :SIGNED_INT16, 0x09,
        :SIGNED_INT32, 0x0a,
        :HALF, 0x10,
        :FLOAT, 0x20,
    )

    CUMemoryType = enum(
        :HOST, 0x01,
        :DEVICE, 0x02,
        :ARRAY, 0x03,
        :UNIFIED, 0x04,
    )

    CUPointerAttribute = enum(
        :CONTEXT, 1,
        :MEMORY_TYPE, 2,
        :DEVICE_POINTER, 3,
        :HOST_POINTER, 4,
    )

    CUJitOption = enum(
        :MAX_REGISTERS, 0,
        :THREADS_PER_BLOCK,
        :WALL_TIME,
        :INFO_LOG_BUFFER,
        :INFO_LOG_BUFFER_SIZE_BYTES,
        :ERROR_LOG_BUFFER,
        :ERROR_LOG_BUFFER_SIZE_BYTES,
        :OPTIMIZATION_LEVEL,
        :TARGET_FROM_CUCONTEXT,
        :TARGET,
        :FALLBACK_STRATEGY,
    )

    CUJitFallBack = enum(
        :PREFER_PTX, 0,
        :PREFER_BINARY,
    )

    CUJitTarget = enum(
        :COMPUTE_10, 0,
        :COMPUTE_11,
        :COMPUTE_12,
        :COMPUTE_13,
        :COMPUTE_20,
        :COMPUTE_21,
    )

    FFI::typedef :int, :enum
    FFI::typedef :int, :CUDevice
    FFI::typedef :pointer, :CUDevicePtr
    FFI::typedef :pointer, :CUContext
    FFI::typedef :pointer, :CUModule
    FFI::typedef :pointer, :CUFunction
    FFI::typedef :pointer, :CUArray
    FFI::typedef :pointer, :CUTexRef
    FFI::typedef :pointer, :CUSurfRef
    FFI::typedef :pointer, :CUEvent
    FFI::typedef :pointer, :CUStream

    def read_int(ptr); ptr.read_int; end
    def read_long(ptr); ptr.read_long; end
    def read_pointer(ptr); ptr.read_pointer; end

    def write_int(ptr); ptr.write_int; end
    def write_long(ptr); ptr.write_long; end
    def write_pointer(ptr, value); ptr.write_pointer(value); end

    alias read_size_t read_long
    alias read_enum read_int
    alias read_cudevice read_int
    alias read_cudeviceptr read_pointer
    alias read_cucontext read_pointer
    alias read_cumodule read_pointer
    alias read_cufunction read_pointer
    alias read_cuarray read_pointer
    alias read_cutexref read_pointer
    alias read_cusurfref read_pointer
    alias read_cuevent read_pointer
    alias read_custream read_pointer

    alias write_size_t write_long
    alias write_enum write_int
    alias write_cudevice write_int
    alias write_cudeviceptr write_pointer
    alias write_cucontext write_pointer
    alias write_cumodule write_pointer
    alias write_cufunction write_pointer
    alias write_cuarray write_pointer
    alias write_cutexref write_pointer
    alias write_cusurfref write_pointer
    alias write_cuevent write_pointer
    alias write_custream write_pointer

    module_function :read_size_t
    module_function :read_enum
    module_function :read_cudevice
    module_function :read_cudeviceptr
    module_function :read_cucontext
    module_function :read_cumodule
    module_function :read_cufunction
    module_function :read_cuarray
    module_function :read_cutexref
    module_function :read_cusurfref
    module_function :read_cuevent
    module_function :read_custream

    module_function :write_size_t
    module_function :write_enum
    module_function :write_cudevice
    module_function :write_cudeviceptr
    module_function :write_cucontext
    module_function :write_cumodule
    module_function :write_cufunction
    module_function :write_cuarray
    module_function :write_cutexref
    module_function :write_cusurfref
    module_function :write_cuevent
    module_function :write_custream


    class CUDevProp < FFI::PrettyStruct
        layout(
            :maxThreadsPerBlock, :int,
            :maxThreadsDim, [:int, 3],
            :maxGridSize, [:int, 3],
            :sharedMemPerBlock, :int,
            :totalConstantMemory, :int,
            :SIMDWidth, :int,
            :memPitch, :int,
            :regsPerBlock, :int,
            :clockRate, :int,
            :textureAlign, :int,
        )
    end

    class CudaMemcpy2D < FFI::PrettyStruct
        layout(
            :srcXInBytes, :size_t,
            :srcY, :size_t,
            :srcMemoryType, CUMemoryType,
            :srcHost, :pointer,
            :srcDevice, :CUDevicePtr,
            :srcArray, :CUArray,
            :srcPitch, :size_t,
            :dstXInBytes, :size_t,
            :dstY, :size_t,
            :dstMemoryType, CUMemoryType,
            :dstHost, :pointer,
            :dstDevice, :CUDevicePtr,
            :dstArray, :CUArray,
            :dstPitch, :size_t,
            :WidthInBytes, :size_t,
            :Height, :size_t,
        )
    end

    class CudaMemcpy3D < FFI::PrettyStruct
        layout(
            :srcXInBytes, :size_t,
            :srcY, :size_t,
            :srcZ, :size_t,
            :srcLOD, :size_t,
            :srcMemoryType, CUMemoryType,
            :srcHost, :pointer,
            :srcDevice, :CUDevicePtr,
            :srcArray, :CUArray,
            :reserved0, :pointer,
            :srcPitch, :size_t,
            :srcHeight, :size_t,
            :dstXInBytes, :size_t,
            :dstY, :size_t,
            :dstZ, :size_t,
            :dstLOD, :size_t,
            :dstMemoryType, CUMemoryType,
            :dstHost, :pointer,
            :dstDevice, :CUDevicePtr,
            :dstArray, :CUArray,
            :reserved1, :pointer,
            :dstPitch, :size_t,
            :dstHeight, :size_t,
            :WidthInBytes, :size_t,
            :Height, :size_t,
            :Depth, :size_t,
        )
    end

    class CudaMemcpy3DPeer < FFI::PrettyStruct
        layout(
            :srcXInBytes, :size_t,
            :srcY, :size_t,
            :srcZ, :size_t,
            :srcLOD, :size_t,
            :srcMemoryType, CUMemoryType,
            :srcHost, :pointer,
            :srcDevice, :CUDevicePtr,
            :srcArray, :CUArray,
            :srcContext, :CUContext,
            :srcPitch, :size_t,
            :srcHeight, :size_t,
            :dstXInBytes, :size_t,
            :dstY, :size_t,
            :dstZ, :size_t,
            :dstLOD, :size_t,
            :dstMemoryType, CUMemoryType,
            :dstHost, :pointer,
            :dstDevice, :CUDevicePtr,
            :dstArray, :CUArray,
            :dstContext, :CUContext,
            :dstPitch, :size_t,
            :dstHeight, :size_t,
            :WidthInBytes, :size_t,
            :Height, :size_t,
            :Depth, :size_t,
        )
    end

    class CudaArrayDescriptor < FFI::PrettyStruct
        layout(
            :Width, :size_t,
            :Height, :size_t,
            :Format, CUArrayFormat,
            :NumChannels, :uint,
        )
    end

    class CudaArray3DDescriptor < FFI::PrettyStruct
        layout(
            :Width, :size_t,
            :Height, :size_t,
            :Depth, :size_t,
            :Format, CUArrayFormat,
            :NumChannels, :uint,
            :Flags, :uint,
        )
    end

    # Initialization.
    attach_function :cuInit, [:uint], :enum

    # CU Version Management.
    attach_function :cuDriverGetVersion, [:pointer], :enum

    # CU Device Management.
    attach_function :cuDeviceComputeCapability, [:pointer, :pointer, :CUDevice], :enum
    attach_function :cuDeviceGet, [:pointer, :int], :enum
    attach_function :cuDeviceGetAttribute, [:pointer, CUDeviceAttribute, :CUDevice], :enum
    attach_function :cuDeviceGetCount, [:pointer], :enum
    attach_function :cuDeviceGetName, [:pointer, :int, :CUDevice], :enum
    attach_function :cuDeviceGetProperties, [:pointer, :CUDevice], :enum
    attach_function :cuDeviceTotalMem, [:pointer, :CUDevice], :enum

    # CU Context Management.
    attach_function :cuCtxCreate, [:pointer, :uint, :CUDevice], :enum
    attach_function :cuCtxDestroy, [:CUContext], :enum
    attach_function :cuCtxGetApiVersion, [:CUContext, :pointer], :enum
    attach_function :cuCtxGetCacheConfig, [:pointer], :enum
    attach_function :cuCtxGetCurrent, [:pointer], :enum
    attach_function :cuCtxGetDevice, [:pointer], :enum
    attach_function :cuCtxGetLimit, [:pointer, CULimit], :enum
    attach_function :cuCtxPopCurrent, [:pointer], :enum
    attach_function :cuCtxPushCurrent, [:CUContext], :enum
    attach_function :cuCtxSetCacheConfig, [CUFunctionCache], :enum
    attach_function :cuCtxSetCurrent, [:CUContext], :enum
    attach_function :cuCtxSetLimit, [CULimit, :size_t], :enum
    attach_function :cuCtxSynchronize, [], :enum
    # Deprecated.
    attach_function :cuCtxAttach, [:pointer, :uint], :enum
    attach_function :cuCtxDetach, [:CUContext], :enum

    # CU Memory Management.
    attach_function :cuArray3DCreate, [:pointer, :pointer], :enum
    attach_function :cuArray3DGetDescriptor, [:pointer, :CUArray], :enum
    attach_function :cuArrayCreate, [:pointer, :pointer], :enum
    attach_function :cuArrayDestroy, [:CUArray], :enum
    attach_function :cuArrayGetDescriptor, [:pointer, :CUArray], :enum
    attach_function :cuMemAlloc, [:pointer, :size_t], :enum
    attach_function :cuMemAllocHost, [:pointer, :size_t], :enum
    attach_function :cuMemAllocPitch, [:pointer, :pointer, :size_t, :size_t, :uint], :enum
    attach_function :cuMemcpy, [:CUDevicePtr, :CUDevicePtr, :size_t], :enum
    attach_function :cuMemcpy2D, [:pointer], :enum
    attach_function :cuMemcpy2DAsync, [:pointer, :CUStream], :enum
    attach_function :cuMemcpy2DUnaligned, [:pointer], :enum
    attach_function :cuMemcpy3D, [:pointer], :enum
    attach_function :cuMemcpy3DAsync, [:pointer, :CUStream], :enum
    attach_function :cuMemcpy3DPeer, [:pointer], :enum
    attach_function :cuMemcpy3DPeerAsync, [:pointer, :CUStream], :enum
    attach_function :cuMemcpyAsync, [:CUDevicePtr, :CUDevicePtr, :size_t, :CUStream], :enum
    attach_function :cuMemcpyAtoA, [:CUArray, :size_t, :CUArray, :size_t, :size_t], :enum
    attach_function :cuMemcpyAtoD, [:CUDevicePtr, :CUArray, :size_t, :size_t], :enum
    attach_function :cuMemcpyAtoH, [:pointer, :CUArray, :size_t, :size_t], :enum
    attach_function :cuMemcpyAtoHAsync, [:pointer, :CUArray, :size_t, :size_t, :CUStream], :enum
    attach_function :cuMemcpyDtoA, [:CUArray, :size_t, :CUDevicePtr, :size_t], :enum
    attach_function :cuMemcpyDtoD, [:CUDevicePtr, :CUDevicePtr, :size_t], :enum
    attach_function :cuMemcpyDtoDAsync, [:CUDevicePtr, :CUDevicePtr, :size_t, :CUStream], :enum
    attach_function :cuMemcpyDtoH, [:pointer, :CUDevicePtr, :size_t], :enum
    attach_function :cuMemcpyDtoHAsync, [:pointer, :CUDevicePtr, :size_t, :CUStream], :enum
    attach_function :cuMemcpyHtoA, [:CUArray, :size_t, :pointer, :size_t], :enum
    attach_function :cuMemcpyHtoAAsync, [:CUArray, :size_t, :pointer, :size_t, :CUStream], :enum
    attach_function :cuMemcpyHtoD, [:CUDevicePtr, :pointer, :size_t], :enum
    attach_function :cuMemcpyHtoDAsync, [:CUDevicePtr, :pointer, :size_t, :CUStream], :enum
    attach_function :cuMemcpyPeer, [:CUDevicePtr, :CUContext, :CUDevicePtr, :CUContext, :size_t], :enum
    attach_function :cuMemcpyPeerAsync, [:CUDevicePtr, :CUContext, :CUDevicePtr, :CUContext, :size_t, :CUStream], :enum
    attach_function :cuMemFree, [:CUDevicePtr], :enum
    attach_function :cuMemFreeHost, [:pointer], :enum
    attach_function :cuMemGetAddressRange, [:pointer, :pointer, :CUDevicePtr], :enum
    attach_function :cuMemGetInfo, [:pointer, :pointer], :enum
    attach_function :cuMemHostAlloc, [:pointer, :size_t, :uint], :enum
    attach_function :cuMemHostGetDevicePointer, [:pointer, :pointer, :uint], :enum
    attach_function :cuMemHostGetFlags, [:pointer, :pointer], :enum
    attach_function :cuMemHostRegister, [:pointer, :size_t, :uint], :enum
    attach_function :cuMemHostUnregister, [:pointer], :enum
    attach_function :cuMemsetD16, [:CUDevicePtr, :ushort, :size_t], :enum
    attach_function :cuMemsetD16Async, [:CUDevicePtr, :ushort, :size_t, :CUStream], :enum
    attach_function :cuMemsetD2D16, [:CUDevicePtr, :size_t, :ushort, :size_t, :size_t], :enum
    attach_function :cuMemsetD2D16Async, [:CUDevicePtr, :size_t, :ushort, :size_t, :size_t, :CUStream], :enum
    attach_function :cuMemsetD2D32, [:CUDevicePtr, :size_t, :uint, :size_t, :size_t], :enum
    attach_function :cuMemsetD2D32Async, [:CUDevicePtr, :size_t, :uint, :size_t, :size_t, :CUStream], :enum
    attach_function :cuMemsetD2D8, [:CUDevicePtr, :size_t, :uchar, :size_t, :size_t], :enum
    attach_function :cuMemsetD2D8Async, [:CUDevicePtr, :size_t, :uchar, :size_t, :size_t, :CUStream], :enum
    attach_function :cuMemsetD32, [:CUDevicePtr, :uint, :size_t], :enum
    attach_function :cuMemsetD32Async, [:CUDevicePtr, :uint, :size_t, :CUStream], :enum
    attach_function :cuMemsetD8, [:CUDevicePtr, :uchar, :size_t], :enum
    attach_function :cuMemsetD8Async, [:CUDevicePtr, :uchar, :size_t, :CUStream], :enum

    # CU Unified Addressing.
    attach_function :cuPointerGetAttribute, [:pointer, CUPointerAttribute, :CUDevicePtr], :enum

    # CU Peer Context Memory Access.
    attach_function :cuCtxDisablePeerAccess, [:CUContext], :enum
    attach_function :cuCtxEnablePeerAccess, [:CUContext, :uint], :enum
    attach_function :cuDeviceCanAccessPeer, [:pointer, :CUDevice, :CUDevice], :enum

    # CU Module Management.
    attach_function :cuModuleGetFunction, [:pointer, :CUModule, :string], :enum
    attach_function :cuModuleGetGlobal, [:pointer, :pointer, :CUModule, :string], :enum
    attach_function :cuModuleGetSurfRef, [:pointer, :CUModule, :string], :enum
    attach_function :cuModuleGetTexRef, [:pointer, :CUModule, :string], :enum
    attach_function :cuModuleLoad, [:pointer, :string], :enum
    attach_function :cuModuleLoadData, [:pointer, :pointer], :enum
    attach_function :cuModuleLoadDataEx, [:pointer, :pointer, :uint, :pointer, :pointer], :enum
    attach_function :cuModuleLoadFatBinary, [:pointer, :pointer], :enum
    attach_function :cuModuleUnload, [:CUModule], :enum

    # CU Execution Control.
    attach_function :cuFuncGetAttribute, [:pointer, CUFunctionAttribute, :CUFunction], :enum
    attach_function :cuFuncSetCacheConfig, [:CUFunction, CUFunctionCache], :enum
    attach_function :cuLaunchKernel, [:CUFunction, :uint, :uint, :uint, :uint, :uint, :uint, :uint, :CUStream, :pointer, :pointer], :enum
    # Deprecated.
    attach_function :cuFuncSetBlockShape, [:CUFunction, :int, :int, :int], :enum
    attach_function :cuFuncSetSharedSize, [:CUFunction, :uint], :enum
    attach_function :cuLaunch, [:CUFunction], :enum
    attach_function :cuLaunchGrid, [:CUFunction, :int, :int], :enum
    attach_function :cuLaunchGridAsync, [:CUFunction, :int, :int, :CUStream], :enum
    attach_function :cuParamSetf, [:CUFunction, :int, :float], :enum
    attach_function :cuParamSeti, [:CUFunction, :int, :uint], :enum
    attach_function :cuParamSetSize, [:CUFunction, :uint], :enum
    attach_function :cuParamSetTexRef, [:CUFunction, :int, :CUTexRef], :enum
    attach_function :cuParamSetv, [:CUFunction, :int, :pointer, :uint], :enum

    # CU Stream Management.
    attach_function :cuStreamCreate, [:pointer, :uint], :enum
    attach_function :cuStreamDestroy, [:CUStream], :enum
    attach_function :cuStreamQuery, [:CUStream], :enum
    attach_function :cuStreamSynchronize, [:CUStream], :enum
    attach_function :cuStreamWaitEvent, [:CUStream, :CUEvent, :uint], :enum

    # CU Event Management.
    attach_function :cuEventCreate, [:pointer, :uint], :enum
    attach_function :cuEventDestroy, [:CUEvent], :enum
    attach_function :cuEventElapsedTime, [:pointer, :CUEvent, :CUEvent], :enum
    attach_function :cuEventQuery, [:CUEvent], :enum
    attach_function :cuEventRecord, [:CUEvent, :CUStream], :enum
    attach_function :cuEventSynchronize, [:CUEvent], :enum

    # CU Texture Reference Management.
    attach_function :cuTexRefGetAddress, [:pointer, :CUTexRef], :enum
    attach_function :cuTexRefGetAddressMode, [:pointer, :CUTexRef, :int], :enum
    attach_function :cuTexRefGetArray, [:pointer, :CUTexRef], :enum
    attach_function :cuTexRefGetFilterMode, [:pointer, :CUTexRef], :enum
    attach_function :cuTexRefGetFlags, [:pointer, :CUTexRef], :enum
    attach_function :cuTexRefGetFormat, [:pointer, :pointer, :CUTexRef], :enum
    attach_function :cuTexRefSetAddress, [:pointer, :CUTexRef, :CUDevicePtr, :size_t], :enum
    attach_function :cuTexRefSetAddress2D, [:CUTexRef, :pointer, :CUDevicePtr, :size_t], :enum
    attach_function :cuTexRefSetAddressMode, [:CUTexRef, :int, CUAddressMode], :enum
    attach_function :cuTexRefSetArray, [:CUTexRef, :CUArray, :uint], :enum
    attach_function :cuTexRefSetFilterMode, [:CUTexRef, CUFilterMode], :enum
    attach_function :cuTexRefSetFlags, [:CUTexRef, :uint], :enum
    attach_function :cuTexRefSetFormat, [:CUTexRef, CUArrayFormat, :int], :enum
    # Deprecated.
    attach_function :cuTexRefCreate, [:pointer], :enum
    attach_function :cuTexRefDestroy, [:CUTexRef], :enum

    # CU Surface Reference Management.
    attach_function :cuSurfRefGetArray, [:pointer, :CUSurfRef], :enum
    attach_function :cuSurfRefSetArray, [:CUSurfRef, :CUArray, :uint], :enum

end # module
end # module
end # module
