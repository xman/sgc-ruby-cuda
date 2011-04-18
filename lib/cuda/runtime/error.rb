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


module SGC
module Cuda

    class CudaStandardError < RuntimeError; end
    class CudaMissingConfigurationError < CudaStandardError; end
    class CudaMemoryAllocationError < CudaStandardError; end
    class CudaInitializationError < CudaStandardError; end
    class CudaLaunchFailureError < CudaStandardError; end
    class CudaPriorLaunchFailureError < CudaStandardError; end # @deprecated
    class CudaLaunchTimeoutError < CudaStandardError; end
    class CudaLaunchOutOfResourcesError < CudaStandardError; end
    class CudaInvalidDeviceFunctionError < CudaStandardError; end
    class CudaInvalidConfigurationError < CudaStandardError; end
    class CudaInvalidDeviceError < CudaStandardError; end
    class CudaInvalidValueError < CudaStandardError; end
    class CudaInvalidPitchValueError < CudaStandardError; end
    class CudaInvalidSymbolError < CudaStandardError; end
    class CudaMapBufferObjectFailedError < CudaStandardError; end
    class CudaUnmapBufferObjectFailedError < CudaStandardError; end
    class CudaInvalidHostPointerError < CudaStandardError; end
    class CudaInvalidDevicePointerError < CudaStandardError; end
    class CudaInvalidTextureError < CudaStandardError; end
    class CudaInvalidTextureBindingError < CudaStandardError; end
    class CudaInvalidChannelDescriptorError < CudaStandardError; end
    class CudaInvalidMemcpyDirectionError < CudaStandardError; end
    class CudaAddressOfConstantError < CudaStandardError; end # @deprecated
    class CudaTextureFetchFailedError  < CudaStandardError; end  # @deprecated
    class CudaTextureNotFoundError < CudaStandardError; end # @deprecated
    class CudaSynchronizationError < CudaStandardError; end # @deprecated
    class CudaInvalidFilterSettingError < CudaStandardError; end
    class CudaInvalidNormSettingError < CudaStandardError; end
    class CudaMixedDeviceExecutionError < CudaStandardError; end # @deprecated
    class CudaCudartUnloadingError < CudaStandardError; end # @deprecated
    class CudaUnknownError < CudaStandardError; end
    class CudaNotYetImplementedError < CudaStandardError; end
    class CudaMemoryValueTooLargeError < CudaStandardError; end # @deprecated
    class CudaInvalidResourceHandleError < CudaStandardError; end
    class CudaNotReadyError < CudaStandardError; end
    class CudaInsufficientDriverError < CudaStandardError; end
    class CudaSetOnActiveProcessError < CudaStandardError; end
    class CudaInvalidSurfaceError < CudaStandardError; end
    class CudaNoDeviceError < CudaStandardError; end
    class CudaECCUncorrectableError < CudaStandardError; end
    class CudaSharedObjectSymbolNotFoundError < CudaStandardError; end
    class CudaSharedObjectInitFailedError < CudaStandardError; end
    class CudaUnsupportedLimitError < CudaStandardError; end
    class CudaDuplicateVariableNameError < CudaStandardError; end
    class CudaDuplicateTextureNameError < CudaStandardError; end
    class CudaDuplicateSurfaceNameError < CudaStandardError; end
    class CudaDevicesUnavailableError < CudaStandardError; end
    class CudaInvalidKernelImageError < CudaStandardError; end
    class CudaNoKernelImageForDeviceError < CudaStandardError; end
    class CudaIncompatibleDriverContextError < CudaStandardError; end
    class CudaPeerAccessAlreadyEnabledError < CudaStandardError; end
    class CudaPeerAccessNotEnabledError < CudaStandardError; end
    class CudaDeviceAlreadyInUseError < CudaStandardError; end
    class CudaProfilerDisabledError < CudaStandardError; end
    class CudaProfilerNotInitializedError < CudaStandardError; end
    class CudaProfilerAlreadyStartedError < CudaStandardError; end
    class CudaProfilerAlreadyStoppedError < CudaStandardError; end
    class CudaStartupFailureError < CudaStandardError; end
    class CudaAPIFailureBaseError < CudaStandardError; end


    # @param [Integer, CudaError] e A CUDA error value or label.
    # @return [String] The error string of _e_.
    def get_error_string(e)
        API::cudaGetErrorString(e)
    end
    module_function :get_error_string


    # @return [Integer] The error value of the last CUDA error.
    def get_last_error
        API::cudaGetLastError
    end
    module_function :get_last_error


    # Return the last CUDA error, but do not reset the error.
    # @return [Integer] The error value of the last CUDA error.
    def peek_at_last_error
        API::cudaPeekAtLastError
    end
    module_function :peek_at_last_error

    # @private
    module Pvt

        def self.handle_error(status, msg = nil)
            status == CUDA_SUCCESS or raise @error_class_by_enum[API::CudaError[status]], API::cudaGetErrorString(status) + " : #{msg}"
            nil
        end


        CUDA_SUCCESS = API::CudaError[:SUCCESS]
        CUDA_ERROR_NOT_READY = API::CudaError[:ERROR_NOT_READY]

        @error_class_by_enum = {
            ERROR_MISSING_CONFIGURATION: CudaMissingConfigurationError,
            ERROR_MEMORY_ALLOCATION: CudaMemoryAllocationError,
            ERROR_INITIALIZATION_ERROR: CudaInitializationError,
            ERROR_LAUNCH_FAILURE: CudaLaunchFailureError,
            ERROR_PRIOR_LAUNCH_FAILURE: CudaPriorLaunchFailureError,
            ERROR_LAUNCH_TIMEOUT: CudaLaunchTimeoutError,
            ERROR_LAUNCH_OUT_OF_RESOURCES: CudaLaunchOutOfResourcesError,
            ERROR_INVALID_DEVICE_FUNCTION: CudaInvalidDeviceFunctionError,
            ERROR_INVALID_CONFIGURATION: CudaInvalidConfigurationError,
            ERROR_INVALID_DEVICE: CudaInvalidDeviceError,
            ERROR_INVALID_VALUE: CudaInvalidValueError,
            ERROR_INVALID_PITCH_VALUE: CudaInvalidPitchValueError,
            ERROR_INVALID_SYMBOL: CudaInvalidSymbolError,
            ERROR_MAP_BUFFER_OBJECT_FAILED: CudaMapBufferObjectFailedError,
            ERROR_UNMAP_BUFFER_OBJECT_FAILED: CudaUnmapBufferObjectFailedError,
            ERROR_INVALID_HOST_POINTER: CudaInvalidHostPointerError,
            ERROR_INVALID_DEVICE_POINTER: CudaInvalidDevicePointerError,
            ERROR_INVALID_TEXTURE: CudaInvalidTextureError,
            ERROR_INVALID_TEXTURE_BINDING: CudaInvalidTextureBindingError,
            ERROR_INVALID_CHANNEL_DESCRIPTOR: CudaInvalidChannelDescriptorError,
            ERROR_INVALID_MEMCPY_DIRECTION: CudaInvalidMemcpyDirectionError,
            ERROR_ADDRESS_OF_CONSTANT: CudaAddressOfConstantError,
            ERROR_TEXTURE_FETCH_FAILED: CudaTextureFetchFailedError,
            ERROR_TEXTURE_NOT_BOUND: CudaTextureNotFoundError,
            ERROR_SYNCHRONIZATION_ERROR: CudaSynchronizationError,
            ERROR_INVALID_FILTER_SETTING: CudaInvalidFilterSettingError,
            ERROR_INVALID_NORM_SETTING: CudaInvalidNormSettingError,
            ERROR_MIXED_DEVICE_EXECUTION: CudaMixedDeviceExecutionError,
            ERROR_CUDART_UNLOADING: CudaCudartUnloadingError,
            ERROR_UNKNOWN: CudaUnknownError,
            ERROR_NOT_YET_IMPLEMENTED: CudaNotYetImplementedError,
            ERROR_MEMORY_VALUE_TOO_LARGE: CudaMemoryValueTooLargeError,
            ERROR_INVALID_RESOURCE_HANDLE: CudaInvalidResourceHandleError,
            ERROR_NOT_READY: CudaNotReadyError,
            ERROR_INSUFFICIENT_DRIVER: CudaInsufficientDriverError,
            ERROR_SET_ON_ACTIVE_PROCESS: CudaSetOnActiveProcessError,
            ERROR_INVALID_SURFACE: CudaInvalidSurfaceError,
            ERROR_NO_DEVICE: CudaNoDeviceError,
            ERROR_ECC_UNCORRECTABLE: CudaECCUncorrectableError,
            ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: CudaSharedObjectSymbolNotFoundError,
            ERROR_SHARED_OBJECT_INIT_FAILED: CudaSharedObjectInitFailedError,
            ERROR_UNSUPPORTED_LIMIT: CudaUnsupportedLimitError,
            ERROR_DUPLICATE_VARIABLE_NAME: CudaDuplicateVariableNameError,
            ERROR_DUPLICATE_TEXTURE_NAME: CudaDuplicateTextureNameError,
            ERROR_DUPLICATE_SURFACE_NAME: CudaDuplicateSurfaceNameError,
            ERROR_DEVICES_UNAVAILABLE: CudaDevicesUnavailableError,
            ERROR_INVALID_KERNEL_IMAGE: CudaInvalidKernelImageError,
            ERROR_NO_KERNEL_IMAGE_FOR_DEVICE: CudaNoKernelImageForDeviceError,
            ERROR_INCOMPATIBLE_DRIVER_CONTEXT: CudaIncompatibleDriverContextError,
            ERROR_PEER_ACCESS_ALREADY_ENABLED: CudaPeerAccessAlreadyEnabledError,
            ERROR_PEER_ACCESS_NOT_ENABLED: CudaPeerAccessNotEnabledError,
            ERROR_DEVICE_ALREADY_IN_USE: CudaDeviceAlreadyInUseError,
            ERROR_PROFILER_DISABLED: CudaProfilerDisabledError,
            ERROR_PROFILER_NOT_INITIALIZED: CudaProfilerNotInitializedError,
            ERROR_PROFILER_ALREADY_STARTED: CudaProfilerAlreadyStartedError,
            ERROR_PROFILER_ALREADY_STOPPED: CudaProfilerAlreadyStoppedError,
            ERROR_STARTUP_FAILURE: CudaStartupFailureError,
            ERROR_API_FAILURE_BASE: CudaAPIFailureBaseError,
        }

    end

end # module
end # module
