#
#   Copyright (c) 2011 Chung Shin Yee
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


module SGC
module CU

    class CUStandardError < RuntimeError; end

    class CUDeviceError < CUStandardError; end
    class CUDeviceNotInitializedError < CUDeviceError; end
    class CUDeviceDeinitializedError < CUDeviceError; end
    class CUNoDeviceError < CUDeviceError; end
    class CUInvalidDeviceError < CUDeviceError; end

    class CUMapError < CUStandardError; end
    class CUMapFailedError < CUMapError; end
    class CUUnMapFailedError < CUMapError; end
    class CUArrayIsMappedError < CUMapError; end
    class CUAlreadyMappedError < CUMapError; end
    class CUNotMappedError < CUMapError; end
    class CUNotMappedAsArrayError < CUMapError; end
    class CUNotMappedAsPointerError < CUMapError; end

    class CUContextError < CUStandardError; end
    class CUInvalidContextError < CUContextError; end
    class CUContextAlreadyCurrentError < CUContextError; end
    class CUContextAlreadyInUseError < CUContextError; end
    class CUUnsupportedLimitError < CUContextError; end
    class CUPrimaryContextActiveError < CUContextError; end
    class CUContextIsDestroyedError < CUContextError; end

    class CULaunchError < CUStandardError; end
    class CULaunchFailedError < CULaunchError; end
    class CULaunchOutOfResourcesError < CULaunchError; end
    class CULaunchTimeoutError < CULaunchError; end
    class CULaunchIncompatibleTexturingError < CULaunchError; end

    class CUParameterError < CUStandardError; end
    class CUInvalidValueError < CUParameterError; end
    class CUInvalidHandleError < CUParameterError; end

    class CUMemoryError < CUStandardError; end
    class CUOutOfMemoryError < CUMemoryError; end

    class CUPeerAccessError < CUStandardError; end
    class CUPeerAccessAlreadyEnabledError < CUPeerAccessError; end
    class CUPeerAccessNotEnabledError < CUPeerAccessError; end

    class CULibraryError < CUStandardError; end
    class CUSharedObjectSymbolNotFoundError < CULibraryError; end
    class CUSharedObjectInitFailedError < CULibraryError; end

    class CUHardwareError < CUStandardError; end
    class CUECCUncorrectableError < CUHardwareError; end

    class CUFileError < CUStandardError; end
    class CUNoBinaryForGPUError < CUFileError; end
    class CUFileNotFoundError < CUFileError; end
    class CUInvalidSourceError < CUFileError; end
    class CUInvalidImageError < CUFileError; end

    class CUReferenceError < CUStandardError; end
    class CUReferenceNotFoundError < CUReferenceError; end

    class CUProfilerError < CUStandardError; end
    class CUProfilerDisabledError < CUProfilerError; end
    class CUProfilerNotInitializedError < CUProfilerError; end
    class CUProfilerAlreadyStartedError < CUProfilerError; end
    class CUProfilerAlreadyStoppedError < CUProfilerError; end

    class CUOtherError < CUStandardError; end
    class CUAlreadyAcquiredError < CUOtherError; end
    class CUNotReadyError < CUOtherError; end
    class CUOperatingSystemError < CUOtherError; end

    class CUUnknownError < CUStandardError; end


    # @private
    module Pvt

        def self.handle_error(status, msg = nil)
            status == CUDA_SUCCESS or raise @error_class_by_enum[API::CUResult[status]], msg
            nil
        end


        CUDA_SUCCESS = API::CUResult[:SUCCESS]
        CUDA_ERROR_NOT_READY = API::CUResult[:ERROR_NOT_READY]

        @error_class_by_enum = {
            ERROR_NOT_INITIALIZED: CUDeviceNotInitializedError,
            ERROR_DEINITIALIZED: CUDeviceDeinitializedError,
            ERROR_NO_DEVICE: CUNoDeviceError,
            ERROR_INVALID_DEVICE: CUInvalidDeviceError,

            ERROR_MAP_FAILED: CUMapFailedError,
            ERROR_UNMAP_FAILED: CUUnMapFailedError,
            ERROR_ARRAY_IS_MAPPED: CUArrayIsMappedError,
            ERROR_ALREADY_MAPPED: CUAlreadyMappedError,
            ERROR_NOT_MAPPED: CUNotMappedError,
            ERROR_NOT_MAPPED_AS_ARRAY: CUNotMappedAsArrayError,
            ERROR_NOT_MAPPED_AS_POINTER: CUNotMappedAsPointerError,

            ERROR_INVALID_CONTEXT: CUInvalidContextError,
            ERROR_CONTEXT_ALREADY_CURRENT: CUContextAlreadyCurrentError,
            ERROR_CONTEXT_ALREADY_IN_USE: CUContextAlreadyInUseError,
            ERROR_UNSUPPORTED_LIMIT: CUUnsupportedLimitError,
            ERROR_PRIMARY_CONTEXT_ACTIVE: CUPrimaryContextActiveError,
            ERROR_CONTEXT_IS_DESTROYED: CUContextIsDestroyedError,

            ERROR_LAUNCH_FAILED: CULaunchFailedError,
            ERROR_LAUNCH_OUT_OF_RESOURCES: CULaunchOutOfResourcesError,
            ERROR_LAUNCH_TIMEOUT: CULaunchTimeoutError,
            ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: CULaunchIncompatibleTexturingError,

            ERROR_INVALID_VALUE: CUInvalidValueError,
            ERROR_INVALID_HANDLE: CUInvalidHandleError,

            ERROR_OUT_OF_MEMORY: CUOutOfMemoryError,

            ERROR_PEER_ACCESS_ALREADY_ENABLED: CUPeerAccessAlreadyEnabledError,
            ERROR_PEER_ACCESS_NOT_ENABLED: CUPeerAccessNotEnabledError,

            ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: CUSharedObjectSymbolNotFoundError,
            ERROR_SHARED_OBJECT_INIT_FAILED: CUSharedObjectInitFailedError,

            ERROR_ECC_UNCORRECTABLE: CUECCUncorrectableError,

            ERROR_NO_BINARY_FOR_GPU: CUNoBinaryForGPUError,
            ERROR_FILE_NOT_FOUND: CUFileNotFoundError,
            ERROR_INVALID_SOURCE: CUInvalidSourceError,
            ERROR_INVALID_IMAGE: CUInvalidImageError,

            ERROR_NOT_FOUND: CUReferenceNotFoundError,

            ERROR_PROFILER_DISABLED: CUProfilerDisabledError,
            ERROR_PROFILER_NOT_INITIALIZED: CUProfilerNotInitializedError,
            ERROR_PROFILER_ALREADY_STARTED: CUProfilerAlreadyStartedError,
            ERROR_PROFILER_ALREADY_STOPPED: CUProfilerAlreadyStoppedError,

            ERROR_ALREADY_ACQUIRED: CUAlreadyAcquiredError,
            ERROR_NOT_READY: CUNotReadyError,
            ERROR_OPERATING_SYSTEM: CUOperatingSystemError,

            ERROR_UNKNOWN: CUUnknownError,
        }

    end

end # module
end # module
