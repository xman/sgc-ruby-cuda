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

require 'ffi'
require 'ffi/prettystruct'
require 'ffi/typedef'
require 'helpers/interface/ienum'
require 'helpers/flags'
require 'helpers/klass'


module SGC
module Cuda
module API

    extend FFI::Library
    ffi_lib "cudart"

    class Enum
        extend SGC::Helper::IEnum
        extend SGC::Helper::FlagsValue

        def self.inherited(subclass)
            subclass.instance_eval %{
                def symbols
                    SGC::Cuda::API::#{SGC::Helper.classname(subclass)}.symbols
                end

                def [](*args)
                    SGC::Cuda::API::#{SGC::Helper.classname(subclass)}[*args]
                end
            }
        end
    end

    CudaError = enum(
        :SUCCESS, 0,
        :ERROR_MISSING_CONFIGURATION, 1,
        :ERROR_MEMORY_ALLOCATION, 2,
        :ERROR_INITIALIZATION_ERROR, 3,
        :ERROR_LAUNCH_FAILURE, 4,
        :ERROR_PRIOR_LAUNCH_FAILURE, 5, # Deprecated as of CUDA 3.1.
        :ERROR_LAUNCH_TIMEOUT, 6,
        :ERROR_LAUNCH_OUT_OF_RESOURCES, 7,
        :ERROR_INVALID_DEVICE_FUNCTION, 8,
        :ERROR_INVALID_CONFIGURATION, 9,
        :ERROR_INVALID_DEVICE, 10,
        :ERROR_INVALID_VALUE, 11,
        :ERROR_INVALID_PITCH_VALUE, 12,
        :ERROR_INVALID_SYMBOL, 13,
        :ERROR_MAP_BUFFER_OBJECT_FAILED, 14,
        :ERROR_UNMAP_BUFFER_OBJECT_FAILED, 15,
        :ERROR_INVALID_HOST_POINTER, 16,
        :ERROR_INVALID_DEVICE_POINTER, 17,
        :ERROR_INVALID_TEXTURE, 18,
        :ERROR_INVALID_TEXTURE_BINDING, 19,
        :ERROR_INVALID_CHANNEL_DESCRIPTOR, 20,
        :ERROR_INVALID_MEMCPY_DIRECTION, 21,
        :ERROR_ADDRESS_OF_CONSTANT, 22, # Deprecated as of CUDA 3.1.
        :ERROR_TEXTURE_FETCH_FAILED, 23, # Deprecated as of CUDA 3.1.
        :ERROR_TEXTURE_NOT_BOUND, 24, # Deprecated as of CUDA 3.1.
        :ERROR_SYNCHRONIZATION_ERROR, 25, # Deprecated as of CUDA 3.1.
        :ERROR_INVALID_FILTER_SETTING, 26,
        :ERROR_INVALID_NORM_SETTING, 27,
        :ERROR_MIXED_DEVICE_EXECUTION, 28, # Deprecated as of CUDA 3.1.
        :ERROR_CUDART_UNLOADING, 29,
        :ERROR_UNKNOWN, 30,
        :ERROR_NOT_YET_IMPLEMENTED, 31, # Deprecated as of CUDA 4.1
        :ERROR_MEMORY_VALUE_TOO_LARGE, 32, # Deprecated as of CUDA 3.1.
        :ERROR_INVALID_RESOURCE_HANDLE, 33,
        :ERROR_NOT_READY, 34,
        :ERROR_INSUFFICIENT_DRIVER, 35,
        :ERROR_SET_ON_ACTIVE_PROCESS, 36,
        :ERROR_INVALID_SURFACE, 37,
        :ERROR_NO_DEVICE, 38,
        :ERROR_ECC_UNCORRECTABLE, 39,
        :ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, 40,
        :ERROR_SHARED_OBJECT_INIT_FAILED, 41,
        :ERROR_UNSUPPORTED_LIMIT, 42,
        :ERROR_DUPLICATE_VARIABLE_NAME, 43,
        :ERROR_DUPLICATE_TEXTURE_NAME, 44,
        :ERROR_DUPLICATE_SURFACE_NAME, 45,
        :ERROR_DEVICES_UNAVAILABLE, 46,
        :ERROR_INVALID_KERNEL_IMAGE, 47,
        :ERROR_NO_KERNEL_IMAGE_FOR_DEVICE, 48,
        :ERROR_INCOMPATIBLE_DRIVER_CONTEXT, 49,
        :ERROR_PEER_ACCESS_ALREADY_ENABLED, 50,
        :ERROR_PEER_ACCESS_NOT_ENABLED, 51,
        :ERROR_DEVICE_ALREADY_IN_USE, 54,
        :ERROR_PROFILER_DISABLED, 55,
        :ERROR_PROFILER_NOT_INITIALIZED, 56,
        :ERROR_PROFILER_ALREADY_STARTED, 57,
        :ERROR_PROFILER_ALREADY_STOPPED, 58,
        :ERROR_STARTUP_FAILURE, 0x7F,
        :ERROR_API_FAILURE_BASE, 10000, # Deprecated as of CUDA 4.1
    )

    CudaDeviceFlags = enum(
        :SCHEDULE_AUTO, 0,
        :SCHEDULE_SPIN, 1,
        :SCHEDULE_YIELD, 2,
        :SCHEDULE_BLOCKING_SYNC, 4,
        :BLOCKING_SYNC, 4, # Deprecated as of CUDA 4.0. Use :SCHEDULE_BLOCKING_SYNC.
        :MAP_HOST, 8,
        :LMEM_RESIZE_TO_MAX, 16,
    )

    CudaEventFlags = enum(
        :DEFAULT, 0,
        :BLOCKING_SYNC, 1,
        :DISABLE_TIMING, 2,
    )

    CudaHostAllocFlags = enum(
        :DEFAULT, 0,
        :PORTABLE, 1,
        :MAPPED, 2,
        :WRITE_COMBINED, 4,
    )

    CudaHostRegisterFlags = enum(
        :DEFAULT, 0,
        :PORTABLE, 1,
        :MAPPED, 2,
    )

    CudaArrayFlags = enum(
        :DEFAULT, 0x00,
        :LAYERED, 0x01,
        :SURFACE_LOAD_STORE, 0x02,
    )

    CudaMemoryType = enum(
        :Host, 1,
        :DEVICE, 2,
    )

    CudaMemcpyKind = enum(
        :HOST_TO_HOST, 0,
        :HOST_TO_DEVICE, 1,
        :DEVICE_TO_HOST, 2,
        :DEVICE_TO_DEVICE, 3,
        :DEFAULT, 4,
    )

    CudaChannelFormatKind = enum(
        :SIGNED, 0,
        :UNSIGNED, 1,
        :FLOAT, 2,
        :None,3,
    )

    CudaFunctionCache = enum(
        :PREFER_NONE, 0,
        :PREFER_SHARED, 1,
        :PREFER_L1, 2,
    )

    CudaLimit = enum(
        :STACK_SIZE, 0x00,
        :PRINTF_FIFO_SIZE, 0x01,
        :MALLOC_HEAP_SIZE, 0x02,
    )

    CudaOutputMode = enum(
        :KEY_VALUE_PAIR, 0x00,
        :CSV, 0x01,
    )

    CudaComputeMode = enum(
        :DEFAULT, 0,
        :EXCLUSIVE, 1,
        :PROHIBITED, 2,
        :EXCLUSIVE_PROCESS, 3,
    )

    CudaSurfaceBoundaryMode = enum(
        :ZERO, 0,
        :CLAMP, 1,
        :TRAP, 2,
    )

    CudaSurfaceFormatMode = enum(
        :FORCED, 0,
        :AUTO, 1,
    )

    CudaTextureAddressMode = enum(
        :WRAP, 0,
        :CLAMP, 1,
        :MIRROR, 2,
        :BORDER, 3,
    )

    CudaTextureFilterMode = enum(
        :POINT, 0,
        :LINEAR, 1,
    )

    CudaTextureReadMode = enum(
        :ELEMENT_TYPE, 0,
        :NORMALIZED_FLOAT, 1,
    )

    FFI::typedef :int, :enum
    FFI::typedef :pointer, :CudaStream
    FFI::typedef :pointer, :CudaEvent

    def read_int(ptr); ptr.read_int; end
    def read_long(ptr); ptr.read_long; end
    def read_pointer(ptr); ptr.read_pointer; end

    def write_int(ptr, value); ptr.write_int(value); end
    def write_long(ptr, value); ptr.write_long(value); end
    def write_pointer(ptr, value); ptr.write_long(value.to_i); end

    alias read_size_t read_long
    alias read_enum read_int
    alias read_cudastream read_pointer
    alias read_cudaevent read_pointer

    alias write_size_t write_long
    alias write_enum write_int
    alias write_cudastream write_pointer
    alias write_cudaevent write_pointer

    module_function :read_size_t
    module_function :read_enum
    module_function :read_cudastream
    module_function :read_cudaevent

    module_function :write_size_t
    module_function :write_enum
    module_function :write_cudastream
    module_function :write_cudaevent


    class Dim3 < FFI::Struct
        layout(
            :array, [:uint, 3],
        )

        alias :init :initialize
        alias :get :[]
        alias :set :[]=
        private :init, :get, :set

        def initialize(x, y = 1, z = 1)
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
            :total_global_mem, :size_t,
            :shared_mem_per_block, :size_t,
            :regs_per_block, :int,
            :warp_size, :int,
            :mem_pitch, :size_t,
            :max_threads_per_block, :int,
            :max_threads_dim, [:int, 3],
            :max_grid_size, [:int, 3],
            :clock_rate, :int,
            :total_const_mem, :size_t,
            :major, :int,
            :minor, :int,
            :texture_alignment, :size_t,
            :texture_pitch_alignment, :size_t,
            :device_overlap, :int, # Deprecated. Use :async_engine_count.
            :multi_processor_count, :int,
            :kernel_exec_timeout_enabled, :int,
            :integrated, :int,
            :can_map_host_memory, :int,
            :compute_mode, :int,
            :max_texture1d, :int,
            :max_texture1d_linear, :int,
            :max_texture2d, [:int, 2],
            :max_texture2d_linear, [:int, 3],
            :max_texture2d_gather, [:int, 2],
            :max_texture3d, [:int, 3],
            :max_texture_cubemap, :int,
            :max_texture1d_layered, [:int, 2],
            :max_texture2d_layered, [:int, 3],
            :max_texture_cubemap_layered, [:int, 2],
            :max_surface1d, :int,
            :max_surface2d, [:int, 2],
            :max_surface3d, [:int, 3],
            :max_surface1d_layered, [:int, 2],
            :max_surface2d_layered, [:int, 3],
            :max_surface_cubemap, :int,
            :max_surface_cubemap_layered, [:int, 2],
            :surface_alignment, :size_t,
            :concurrent_kernels, :int,
            :ecc_enabled, :int,
            :pci_bus_id, :int,
            :pci_device_id, :int,
            :pci_domain_id, :int,
            :tcc_driver, :int,
            :async_engine_count, :int,
            :unified_addressing, :int,
            :memory_clock_rate, :int,
            :memory_bus_width, :int,
            :l2_cache_size, :int,
            :max_threads_per_multi_processor, :int,
        )
    end

    class CudaFunctionAttributes < FFI::PrettyStruct
        layout(
            :shared_size_bytes, :size_t,
            :const_size_bytes, :size_t,
            :local_size_bytes, :size_t,
            :max_threads_per_block, :int,
            :num_regs, :int,
            :ptx_version, :int,
            :binary_version, :int,
        )
    end

    class CudaPointerAttributes < FFI::PrettyStruct
        layout(
            :memory_type, CudaMemoryType,
            :device, :int,
            :device_pointer, :pointer,
            :host_pointer, :pointer,
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
            :src_array, :pointer,
            :src_pos, CudaPos,
            :src_ptr, CudaPitchedPtr,
            :dst_array, :pointer,
            :dst_pos, CudaPos,
            :dst_ptr, CudaPitchedPtr,
            :extent, CudaExtent,
            :kind, CudaMemcpyKind,
        )
    end

    class CudaMemcpy3DPeerParms < FFI::PrettyStruct
        layout(
            :src_array, :pointer,
            :src_pos, CudaPos,
            :src_ptr, CudaPitchedPtr,
            :src_device, :int,
            :dst_array, :pointer,
            :dst_pos, CudaPos,
            :dst_ptr, CudaPitchedPtr,
            :dst_device, :int,
            :extent, CudaExtent,
        )
    end

    class TextureReference < FFI::PrettyStruct
        layout(
            :normalized, :int,
            :filter_mode, CudaTextureFilterMode,
            :address_mode, [CudaTextureAddressMode, 3],
            :channel_desc, CudaChannelFormatDesc,
            :srgb, :int,
            :__cuda_reserved, [:int, 15],
        )
    end

    class SurfaceReference < FFI::PrettyStruct
        layout(
            :channel_desc, CudaChannelFormatDesc,
        )
    end

    # CUDA Version Management.
    attach_function :cudaDriverGetVersion, [:pointer], :enum
    attach_function :cudaRuntimeGetVersion, [:pointer], :enum

    # CUDA Error Handling.
    attach_function :cudaGetErrorString, [CudaError], :string
    attach_function :cudaGetLastError, [], :enum
    attach_function :cudaPeekAtLastError, [], :enum

    # CUDA Device Management.
    attach_function :cudaChooseDevice, [:pointer, :pointer], :enum
    attach_function :cudaDeviceGetCacheConfig, [:pointer], :enum
    attach_function :cudaDeviceGetLimit, [:pointer, CudaLimit], :enum
    attach_function :cudaDeviceReset, [], :enum
    attach_function :cudaDeviceSetCacheConfig, [CudaFunctionCache], :enum
    attach_function :cudaDeviceSetLimit, [CudaLimit, :size_t], :enum
    attach_function :cudaDeviceSynchronize, [], :enum
    attach_function :cudaGetDevice, [:pointer], :enum
    attach_function :cudaGetDeviceCount, [:pointer], :enum
    attach_function :cudaGetDeviceProperties, [:pointer, :int], :enum
    attach_function :cudaSetDevice, [:int], :enum
    attach_function :cudaSetDeviceFlags, [:uint], :enum
    attach_function :cudaSetValidDevices, [:pointer, :int], :enum

    # CUDA Thread Management.
    # Deprecated.
    attach_function :cudaThreadExit, [], :enum
    attach_function :cudaThreadGetCacheConfig, [:pointer], :enum
    attach_function :cudaThreadGetLimit, [:pointer, CudaLimit], :enum
    attach_function :cudaThreadSetCacheConfig, [CudaFunctionCache], :enum
    attach_function :cudaThreadSetLimit, [CudaLimit, :size_t], :enum
    attach_function :cudaThreadSynchronize, [], :enum

    # CUDA Memory Management.
    attach_function :cudaFree, [:pointer], :enum
    attach_function :cudaFreeArray, [:pointer], :enum
    attach_function :cudaFreeHost, [:pointer], :enum
    attach_function :cudaGetSymbolAddress, [:pointer, :string], :enum
    attach_function :cudaGetSymbolSize, [:pointer, :string], :enum
    attach_function :cudaHostAlloc, [:pointer, :size_t, :uint], :enum
    attach_function :cudaHostGetDevicePointer, [:pointer, :pointer, :uint], :enum
    attach_function :cudaHostGetFlags, [:pointer, :pointer], :enum
    attach_function :cudaHostRegister, [:pointer, :size_t, :uint], :enum
    attach_function :cudaHostUnregister, [:pointer], :enum
    attach_function :cudaMalloc, [:pointer, :size_t], :enum
    attach_function :cudaMalloc3D, [:pointer, CudaExtent.by_value], :enum
    attach_function :cudaMalloc3DArray, [:pointer, :pointer, CudaExtent.by_value, :uint], :enum
    attach_function :cudaMallocArray, [:pointer, :pointer, :size_t, :size_t, :uint], :enum
    attach_function :cudaMallocHost, [:pointer, :size_t], :enum
    attach_function :cudaMallocPitch, [:pointer, :pointer, :size_t, :size_t], :enum
    attach_function :cudaMemcpy, [:pointer, :pointer, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpy2D, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpy2DArrayToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpy2DAsync, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpy2DFromArray, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpy2DFromArrayAsync, [:pointer, :size_t, :pointer, :size_t, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpy2DToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpy2DToArrayAsync, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpy3D, [:pointer], :enum
    attach_function :cudaMemcpy3DAsync, [:pointer, :CudaStream], :enum
    attach_function :cudaMemcpy3DPeer, [:pointer], :enum
    attach_function :cudaMemcpy3DPeerAsync, [:pointer, :CudaStream], :enum
    attach_function :cudaMemcpyArrayToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpyAsync, [:pointer, :pointer, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpyFromArray, [:pointer, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpyFromArrayAsync, [:pointer, :pointer, :size_t, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpyFromSymbol, [:pointer, :string, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpyFromSymbolAsync, [:pointer, :string, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpyPeer, [:pointer, :int, :pointer, :int, :size_t], :enum
    attach_function :cudaMemcpyPeerAsync, [:pointer, :int, :pointer, :int, :size_t, :CudaStream], :enum
    attach_function :cudaMemcpyToArray, [:pointer, :size_t, :size_t, :pointer, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpyToArrayAsync, [:pointer, :size_t, :size_t, :pointer, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemcpyToSymbol, [:string, :pointer, :size_t, :size_t, CudaMemcpyKind], :enum
    attach_function :cudaMemcpyToSymbolAsync, [:string, :pointer, :size_t, :size_t, CudaMemcpyKind, :CudaStream], :enum
    attach_function :cudaMemGetInfo, [:pointer, :pointer], :enum
    attach_function :cudaMemset, [:pointer, :int, :size_t], :enum
    attach_function :cudaMemset2D, [:pointer, :size_t, :int, :size_t, :size_t], :enum
    attach_function :cudaMemset2DAsync, [:pointer, :size_t, :int, :size_t, :size_t, :CudaStream], :enum
    attach_function :cudaMemset3D, [CudaPitchedPtr.by_value, :int, CudaExtent.by_value], :enum
    attach_function :cudaMemset3DAsync, [CudaPitchedPtr.by_value, :int, CudaExtent.by_value, :CudaStream], :enum
    attach_function :cudaMemsetAsync, [:pointer, :int, :size_t, :CudaStream], :enum
    # attach_function :make_cudaExtent, [:size_t, :size_t, :size_t], CudaExtent
    # attach_function :make_cudaPitchedPtr, [:pointer, :size_t, :size_t, :size_t], CudaPitchedPtr
    # attach_function :make_cudaPos, [:size_t, :size_t, :size_t], CudaPos

    def make_cudaExtent(w, h, d)
        e = CudaExtent.new
        e[:width], e[:height], e[:depth] = w, h, d
        e
    end
    module_function :make_cudaExtent

    def make_cudaPitchedPtr(d, p, xsz, ysz)
        s = CudaPitchedPtr.new
        s[:ptr] = d
        s[:pitch] = p
        s[:xsize] = xsz
        s[:ysize] = ysz
        s
    end
    module_function :make_cudaPitchedPtr

    def make_cudaPos(x, y, z)
        p = CudaPos.new
        p[:x] = x
        p[:y] = y
        p[:z] = z
        p
    end
    module_function :make_cudaPos

    # CUDA Unified Addressing.
    attach_function :cudaPointerGetAttributes, [:pointer, :pointer], :enum

    # CUDA Peer Device Memory Access.
    attach_function :cudaDeviceCanAccessPeer, [:pointer, :int, :int], :enum
    attach_function :cudaDeviceDisablePeerAccess, [:int], :enum
    attach_function :cudaDeviceEnablePeerAccess, [:int, :uint], :enum

    # CUDA Execution Control.
    attach_function :cudaConfigureCall, [Dim3.by_value, Dim3.by_value, :size_t, :CudaStream], :enum
    attach_function :cudaFuncGetAttributes, [:pointer, :string], :enum
    attach_function :cudaFuncSetCacheConfig, [:string, CudaFunctionCache], :enum
    attach_function :cudaLaunch, [:string], :enum
    attach_function :cudaSetDoubleForDevice, [:pointer], :enum
    attach_function :cudaSetDoubleForHost, [:pointer], :enum
    attach_function :cudaSetupArgument, [:pointer, :size_t, :size_t], :enum

    # CUDA Stream Management.
    attach_function :cudaStreamCreate, [:pointer], :enum
    attach_function :cudaStreamDestroy, [:CudaStream], :enum
    attach_function :cudaStreamQuery, [:CudaStream], :enum
    attach_function :cudaStreamSynchronize, [:CudaStream], :enum
    attach_function :cudaStreamWaitEvent, [:CudaStream, :CudaEvent, :uint], :enum

    # CUDA Event Management.
    attach_function :cudaEventCreate, [:pointer], :enum
    attach_function :cudaEventCreateWithFlags, [:pointer, :uint], :enum
    attach_function :cudaEventDestroy, [:CudaEvent], :enum
    attach_function :cudaEventElapsedTime, [:pointer, :CudaEvent, :CudaEvent], :enum
    attach_function :cudaEventQuery, [:CudaEvent], :enum
    attach_function :cudaEventRecord, [:CudaEvent, :CudaStream], :enum
    attach_function :cudaEventSynchronize, [:CudaEvent], :enum

    # CUDA Texture Reference Management.
    attach_function :cudaBindTexture, [:pointer, :pointer, :pointer, :pointer, :size_t], :enum
    attach_function :cudaBindTexture2D, [:pointer, :pointer, :pointer, :pointer, :size_t, :size_t, :size_t], :enum
    attach_function :cudaBindTextureToArray, [:pointer, :pointer, :pointer], :enum
    attach_function :cudaCreateChannelDesc, [:int, :int, :int, :int, CudaChannelFormatKind], CudaChannelFormatDesc.by_value
    attach_function :cudaGetChannelDesc, [:pointer, :pointer], :enum
    attach_function :cudaGetTextureAlignmentOffset, [:pointer, :pointer], :enum
    attach_function :cudaUnbindTexture, [:pointer], :enum
    # Deprecated as of CUDA 4.1.
    attach_function :cudaGetTextureReference, [:pointer, :string], :enum

    # CUDA Surface Reference Management.
    attach_function :cudaBindSurfaceToArray, [:pointer, :pointer, :pointer], :enum
    # Deprecated as of CUDA 4.1.
    attach_function :cudaGetSurfaceReference, [:pointer, :string], :enum

end # module
end # module
end # module
