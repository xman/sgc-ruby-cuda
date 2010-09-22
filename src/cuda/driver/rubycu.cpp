/*
# Copyright (c) 2010 Chung Shin Yee
#
#       shinyee@speedgocomputing.com
#       http://www.speedgocomputing.com
#       http://github.com/xman/sgc-ruby-cuda
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
*/

#include <cstring>
#include "ruby.h"
#include "cuda.h"

// {{{ SGC Ruby modules.
static VALUE rb_mSGC;
static VALUE rb_mCU;
// }}}

// {{{ CUDA Ruby classes.
static VALUE rb_cCUDevice;
static VALUE rb_cCUContext;
static VALUE rb_cCUContextFlags;
static VALUE rb_cCULimit;
static VALUE rb_cCUModule;
static VALUE rb_cCUFunction;
static VALUE rb_cCUFunctionAttribute;
static VALUE rb_cCUFunctionCache;
static VALUE rb_cCUDevicePtr;
static VALUE rb_cCUDeviceAttribute;
static VALUE rb_cCUComputeMode;
static VALUE rb_cCUStream;
static VALUE rb_cCUEvent;
static VALUE rb_cCUResult;
// }}}

// {{{ SGC Ruby classes.
static VALUE rb_eCUStandardError;

static VALUE rb_eCUDeviceError;
static VALUE rb_eCUDeviceNotInitializedError;
static VALUE rb_eCUDeviceDeinitializedError;
static VALUE rb_eCUNoDeviceError;
static VALUE rb_eCUInvalidDeviceError;

static VALUE rb_eCUMapError;
static VALUE rb_eCUMapFailedError;
static VALUE rb_eCUUnMapFailedError;
static VALUE rb_eCUArrayIsMappedError;
static VALUE rb_eCUAlreadyMappedError;
static VALUE rb_eCUNotMappedError;
static VALUE rb_eCUNotMappedAsArrayError;
static VALUE rb_eCUNotMappedAsPointerError;

static VALUE rb_eCUContextError;
static VALUE rb_eCUInvalidContextError;
static VALUE rb_eCUContextAlreadyCurrentError;
static VALUE rb_eCUUnsupportedLimitError;

static VALUE rb_eCULaunchError;
static VALUE rb_eCULaunchFailedError;
static VALUE rb_eCULaunchOutOfResourcesError;
static VALUE rb_eCULaunchTimeoutError;
static VALUE rb_eCULaunchIncompatibleTexturingError;

static VALUE rb_eCUBitWidthError;
static VALUE rb_eCUPointerIs64BitError;
static VALUE rb_eCUSizeIs64BitError;

static VALUE rb_eCUParameterError;
static VALUE rb_eCUInvalidValueError;
static VALUE rb_eCUInvalidHandleError;

static VALUE rb_eCUMemoryError;
static VALUE rb_eCUOutOfMemoryError;

static VALUE rb_eCULibraryError;
static VALUE rb_eCUSharedObjectSymbolNotFoundError;
static VALUE rb_eCUSharedObjectInitFailedError;

static VALUE rb_eCUHardwareError;
static VALUE rb_eCUECCUncorrectableError;

static VALUE rb_eCUFileError;
static VALUE rb_eCUNoBinaryForGPUError;
static VALUE rb_eCUFileNotFoundError;
static VALUE rb_eCUInvalidSourceError;
static VALUE rb_eCUInvalidImageError;

static VALUE rb_eCUReferenceError;
static VALUE rb_eCUReferenceNotFoundError;

static VALUE rb_eCUOtherError;
static VALUE rb_eCUAlreadyAcquiredError;
static VALUE rb_eCUNotReadyError;

static VALUE rb_eCUUnknownError;

static VALUE rb_cMemoryBuffer;
static VALUE rb_cInt32Buffer;
static VALUE rb_cInt64Buffer;
static VALUE rb_cFloat32Buffer;
static VALUE rb_cFloat64Buffer;
// }}}

// {{{ SGC C/C++ structures.
typedef struct {
    size_t size;
    char* p;
} MemoryBuffer;

template <typename TElement>
struct TypedBuffer : public MemoryBuffer {};

typedef struct TypedBuffer<int>    Int32Buffer;
typedef struct TypedBuffer<long>   Int64Buffer;
typedef struct TypedBuffer<float>  Float32Buffer;
typedef struct TypedBuffer<double> Float64Buffer;
// }}}

// {{{ Function prototypes.
static VALUE device_ptr_alloc(VALUE klass);
static VALUE device_ptr_initialize(int argc, VALUE* argv, VALUE self);
// }}}

// {{{ SGC helpers.
template <typename T>
static void generic_free(void* p)
{
    delete static_cast<T*>(p);
}

template <typename T>
static VALUE to_rb(T v);

template <>
VALUE to_rb(int v)
{
    return INT2FIX(v);
}

template <>
VALUE to_rb(long v)
{
    return LONG2NUM(v);
}

template <>
VALUE to_rb(float v)
{
    return DBL2NUM(static_cast<double>(v));
}

template <>
VALUE to_rb(double v)
{
    return DBL2NUM(v);
}

template <typename T>
static T to_ctype(VALUE v);

template <>
int to_ctype<int>(VALUE v)
{
    return NUM2INT(v);
}

template <>
unsigned int to_ctype<unsigned int>(VALUE v)
{
    return NUM2UINT(v);
}

template <>
long to_ctype<long>(VALUE v)
{
    return NUM2LONG(v);
}

template <>
unsigned long to_ctype<unsigned long>(VALUE v)
{
    return NUM2ULONG(v);
}

template <>
float to_ctype<float>(VALUE v)
{
    return static_cast<float>(NUM2DBL(v));
}

template <>
double to_ctype(VALUE v)
{
    return NUM2DBL(v);
}

// in  ary[0]: Class contains class constants.
// in  ary[1]: Constant to match.
// out ary[2]: Label matches with constant.
static VALUE class_const_match(VALUE current_label, VALUE* ary)
{
    const VALUE& rb_class_const = ary[0];
    const VALUE& constant_value = ary[1];
    VALUE& label = ary[2];
    VALUE v = rb_funcall(rb_class_const, rb_intern("const_get"), 1, current_label);
    if (FIX2INT(v) == FIX2INT(constant_value)) {
        label = current_label;
        return Qtrue;
    }
    return Qfalse;
}

#define RAISE_CU_STD_ERROR_FORMATTED(status, format, ...) rb_raise(rb_hash_aref(rb_error_class_by_enum, INT2FIX(status)), "%s:%d " format, __FILE__, __LINE__, __VA_ARGS__)
#define RAISE_CU_STD_ERROR(status, message) RAISE_CU_STD_ERROR_FORMATTED(status, "%s", message)
// }}}

// {{{ SGC Ruby data.
static VALUE rb_error_class_by_enum;
// }}}


// {{{ CUdevice
static VALUE device_get_count(VALUE klass)
{
    int count;
    CUresult status = cuDeviceGetCount(&count);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get device count.");
    }
    return INT2FIX(count);
}

static VALUE device_alloc(VALUE klass)
{
    CUdevice* p = new CUdevice;
    return Data_Wrap_Struct(klass, 0, generic_free<CUdevice>, p);
}

static VALUE device_initialize(int argc, VALUE* argv, VALUE self)
{
    return self;
}

static VALUE device_get(VALUE self, VALUE num)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int i = FIX2INT(num);
    CUresult status = cuDeviceGet(p, i);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get device %d.", i);
    }
    return self;
}

static VALUE device_get_name(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    char name[256];
    CUresult status = cuDeviceGetName(name, 256, *p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get device name.");
    }
    return rb_str_new2(name);
}

static VALUE device_compute_capability(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int major;
    int minor;
    CUresult status = cuDeviceComputeCapability(&major, &minor, *p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to query device compute capability.");
    }
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("major")), INT2FIX(major));
    rb_hash_aset(h, ID2SYM(rb_intern("minor")), INT2FIX(minor));
    return h;
}

static VALUE device_get_attribute(VALUE self, VALUE attribute)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int v;
    CUresult status = cuDeviceGetAttribute(&v, static_cast<CUdevice_attribute>(FIX2INT(attribute)), *p);
    if (status != CUDA_SUCCESS) {
        VALUE attributes = rb_funcall(rb_cCUDeviceAttribute, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCUDeviceAttribute, attribute, Qnil };
        rb_block_call(attributes, rb_intern("find"), 0, NULL, (VALUE(*)(ANYARGS))class_const_match, (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to query device attribute: %s.", rb_id2name(SYM2ID(ary[2])));
    }
    return INT2FIX(v);
}

static VALUE device_total_mem(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    unsigned int nbytes;
    CUresult status = cuDeviceTotalMem(&nbytes, *p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get device total amount of memory available.");
    }
    long n = static_cast<long>(nbytes);
    return LONG2NUM(n);
}
// }}}


// {{{ CUcontext
static VALUE context_alloc(VALUE klass)
{
    CUcontext* p = new CUcontext;
    return Data_Wrap_Struct(klass, 0, generic_free<CUcontext>, p);
}

static VALUE context_initialize(int argc, VALUE* argv, VALUE self)
{
    return self;
}

static VALUE context_create(VALUE self, VALUE flags, VALUE rb_device)
{
    CUcontext* pcontext;
    CUdevice* pdevice;
    Data_Get_Struct(self, CUcontext, pcontext);
    Data_Get_Struct(rb_device, CUdevice, pdevice);
    CUresult status = cuCtxCreate(pcontext, FIX2UINT(flags), *pdevice);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to create context: flags = %x.", FIX2UINT(flags));
    }
    return self;
}

static VALUE context_destroy(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    CUresult status = cuCtxDestroy(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to destroy context.");
    }
    return Qnil;
}

static VALUE context_attach(int argc, VALUE* argv, VALUE self)
{
    CUcontext* p;
    unsigned int flags = 0;
    Data_Get_Struct(self, CUcontext, p);
    if (argc == 1) {
        flags = FIX2UINT(argv[0]);
    }
    CUresult status = cuCtxAttach(p, flags);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to attach context: flags = %x.", flags);
    }
    return self;
}

static VALUE context_detach(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    CUresult status = cuCtxDetach(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to detach context.");
    }
    return Qnil;
}

static VALUE context_push_current(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    CUresult status = cuCtxPushCurrent(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to push this context.");
    }
    return self;
}

static VALUE context_get_device(VALUE klass, VALUE device)
{
    CUdevice* pdevice;
    Data_Get_Struct(device, CUdevice, pdevice);
    CUresult status = cuCtxGetDevice(pdevice);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get current context's device.");
    }
    return Qnil;
}

static VALUE context_get_limit(VALUE klass, VALUE limit)
{
    CUlimit l = static_cast<CUlimit>(FIX2UINT(limit));
    size_t v = 0;
    CUresult status = cuCtxGetLimit(&v, l);
    if (status != CUDA_SUCCESS) {
        VALUE limits = rb_funcall(rb_cCULimit, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCULimit, limit, Qnil };
        rb_block_call(limits, rb_intern("find"), 0, NULL, (VALUE(*)(ANYARGS))class_const_match, (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get context limit: %s.", rb_id2name(SYM2ID(ary[2])));
    }
    return SIZET2NUM(v);
}

static VALUE context_set_limit(VALUE klass, VALUE limit, VALUE value)
{
    CUlimit l = static_cast<CUlimit>(FIX2UINT(limit));
    size_t v = NUM2SIZET(value);
    CUresult status = cuCtxSetLimit(l, v);
    if (status != CUDA_SUCCESS) {
        VALUE limits = rb_funcall(rb_cCULimit, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCULimit, limit, Qnil };
        rb_block_call(limits, rb_intern("find"), 0, NULL, (VALUE(*)(ANYARGS))class_const_match, (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set context limit: %s to %u.", rb_id2name(SYM2ID(ary[2])), v);
    }
    return Qnil;
}

static VALUE context_pop_current(VALUE klass, VALUE context)
{
    CUcontext* pcontext;
    Data_Get_Struct(context, CUcontext, pcontext);
    CUresult status = cuCtxPopCurrent(pcontext);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to pop current context.");
    }
    return Qnil;
}

static VALUE context_synchronize(VALUE klass)
{
    CUresult status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to synchronize this context.");
    }
    return Qnil;
}
// }}}


// {{{ CUmodule
static VALUE module_alloc(VALUE klass)
{
    CUmodule* p = new CUmodule;
    return Data_Wrap_Struct(klass, 0, generic_free<CUmodule>, p);
}

static VALUE module_initialize(int argc, VALUE* argv, VALUE self)
{
    return self;
}

static VALUE module_load(VALUE self, VALUE str)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    CUresult status = cuModuleLoad(p, StringValuePtr(str));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to load module: %s.", StringValuePtr(str));
    }
    return self;
}

static VALUE module_unload(VALUE self)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    CUresult status = cuModuleUnload(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to unload module.");
    }
    return self;
}

static VALUE module_get_function(VALUE self, VALUE str)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    CUfunction* pfunc = new CUfunction;
    CUresult status = cuModuleGetFunction(pfunc, *p, StringValuePtr(str));
    if (status != CUDA_SUCCESS) {
        delete pfunc;
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get module function: %s.", StringValuePtr(str));
    }
    return Data_Wrap_Struct(rb_cCUFunction, 0, generic_free<CUfunction>, pfunc);
}

static VALUE module_get_global(VALUE self, VALUE str)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    VALUE rb_devptr = device_ptr_alloc(rb_cCUDevicePtr);
    device_ptr_initialize(0, NULL, rb_devptr);
    CUdeviceptr* pdevptr;
    Data_Get_Struct(rb_devptr, CUdeviceptr, pdevptr);
    unsigned int nbytes;
    CUresult status = cuModuleGetGlobal(pdevptr, &nbytes, *p, StringValuePtr(str));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get module global: %s.", StringValuePtr(str));
    }
    return rb_ary_new3(2, rb_devptr, LONG2NUM(nbytes));
}
// }}}


// {{{ CUdeviceptr
static VALUE device_ptr_alloc(VALUE klass)
{
    CUdeviceptr* p = new CUdeviceptr;
    return Data_Wrap_Struct(klass, 0, generic_free<CUdeviceptr>, p);
}

static VALUE device_ptr_initialize(int argc, VALUE* argv, VALUE self)
{
    CUdeviceptr* p;
    Data_Get_Struct(self, CUdeviceptr, p);
    *p = static_cast<CUdeviceptr>(0);
    return self;
}

static VALUE device_ptr_mem_alloc(VALUE self, VALUE nbytes)
{
    CUdeviceptr* p;
    Data_Get_Struct(self, CUdeviceptr, p);
    size_t n = NUM2SIZET(nbytes);
    cuMemAlloc(p, n);
    return self;
}

static VALUE device_ptr_mem_free(VALUE self)
{
    CUdeviceptr* p;
    Data_Get_Struct(self, CUdeviceptr, p);
    cuMemFree(*p);
    return self;
}
// }}}


// {{{ CUfunction
static VALUE function_alloc(VALUE klass)
{
    CUfunction* p = new CUfunction;
    return Data_Wrap_Struct(klass, 0, generic_free<CUfunction>, p);
}

static VALUE function_initialize(int argc, VALUE* argv, VALUE self)
{
    return self;
}

static VALUE function_set_param(int argc, VALUE* argv, VALUE self)
{
    #define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

    int offset = 0;
    CUfunction* pfunc;
    Data_Get_Struct(self, CUfunction, pfunc);

    for (int i = 0; i < argc; ++i) {
        if (CLASS_OF(argv[i]) == rb_cCUDevicePtr) {
            CUdeviceptr* p;
            void* vp = NULL;
            Data_Get_Struct(argv[i], CUdeviceptr, p);
            vp = (void*)(size_t)(*p);
            ALIGN_UP(offset, __alignof(vp));
            cuParamSetv(*pfunc, offset, &vp, sizeof(vp));
            offset += sizeof(vp);
        } else if (CLASS_OF(argv[i]) == rb_cFixnum) {
            int num = FIX2INT(argv[i]);
            ALIGN_UP(offset, __alignof(num));
            cuParamSeti(*pfunc, offset, num);
            offset += sizeof(int);
        } else if (CLASS_OF(argv[i]) == rb_cFloat) {
            float num = static_cast<float>(NUM2DBL(argv[i]));
            ALIGN_UP(offset, __alignof(num));
            cuParamSetf(*pfunc, offset, num);
            offset += sizeof(float);
        } else {
            rb_raise(rb_eRuntimeError, "Invalid parameters to CUFunction set.");
        }
    }

    cuParamSetSize(*pfunc, offset);
    return self;
}

static VALUE function_set_block_shape(int argc, VALUE* argv, VALUE self)
{
    if (argc <= 0 || argc > 3) {
        rb_raise(rb_eRuntimeError, "Invalid number of parameters to set block shape. Expect 1 to 3 integers.");
    }

    CUfunction* pfunc;
    Data_Get_Struct(self, CUfunction, pfunc);

    int xdim = FIX2INT(argv[0]);
    int ydim = 1;
    int zdim = 1;

    if (argc >= 2) {
        ydim = FIX2INT(argv[1]);
    }
    if (argc >= 3) {
        zdim = FIX2INT(argv[2]);
    }

    cuFuncSetBlockShape(*pfunc, xdim, ydim, zdim);
    return self;
}

static VALUE function_set_shared_size(VALUE self, VALUE nbytes)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    cuFuncSetSharedSize(*p, NUM2UINT(nbytes));
    return self;
}

static VALUE function_launch(VALUE self)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    cuLaunch(*p);
    return self;
}

static VALUE function_launch_grid(int argc, VALUE* argv, VALUE self)
{
    if (argc <= 0 || argc > 2) {
        rb_raise(rb_eRuntimeError, "Invalid number of parameters to launch grid. Expect 1 to 2 integers.");
    }

    CUfunction* pfunc;
    Data_Get_Struct(self, CUfunction, pfunc);

    int xdim = FIX2INT(argv[0]);
    int ydim = 1;

    if (argc >= 2) {
        ydim = FIX2INT(argv[1]);
    }

    cuLaunchGrid(*pfunc, xdim, ydim);
    return self;
}

static VALUE function_get_attribute(VALUE self, VALUE attribute)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    int v;
    cuFuncGetAttribute(&v, static_cast<CUfunction_attribute>(FIX2INT(attribute)), *p);
    return INT2FIX(v);
}

static VALUE function_set_cache_config(VALUE self, VALUE config)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    cuFuncSetCacheConfig(*p, static_cast<CUfunc_cache>(FIX2UINT(config)));
    return self;
}
// }}}


// {{{ CUstream
static VALUE stream_alloc(VALUE klass)
{
    CUstream* p = new CUstream;
    return Data_Wrap_Struct(klass, 0, generic_free<CUstream>, p);
}

static VALUE stream_initialize(VALUE self)
{
    return self;
}

static VALUE stream_create(VALUE self, VALUE flags)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    CUresult status = cuStreamCreate(p, FIX2UINT(flags));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to create stream: flags = %x", FIX2UINT(flags));
    }
    return self;
}

static VALUE stream_destroy(VALUE self)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    CUresult status = cuStreamDestroy(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to destroy stream.");
    }
    return Qnil;
}

static VALUE stream_query(VALUE self)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    CUresult status = cuStreamQuery(*p);
    if (status == CUDA_SUCCESS) {
        return Qtrue;
    } else if (status == CUDA_ERROR_NOT_READY) {
        return Qfalse;
    } else {
        RAISE_CU_STD_ERROR(status, "Failed to query stream.");
    }
}

static VALUE stream_synchronize(VALUE self)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    CUresult status = cuStreamSynchronize(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to synchronize stream.");
    }
    return self;
}
// }}}


// {{{ CUevent
static VALUE event_alloc(VALUE klass)
{
    CUevent* p = new CUevent;
    return Data_Wrap_Struct(klass, 0, generic_free<CUevent>, p);
}

static VALUE event_initialize(VALUE self)
{
    return self;
}

static VALUE event_create(VALUE self, VALUE flags)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    cuEventCreate(p, FIX2UINT(flags));
    return self;
}

static VALUE event_destroy(VALUE self)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    cuEventDestroy(*p);
    return Qnil;
}

static VALUE event_query(VALUE self)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    CUresult status = cuEventQuery(*p);
    if (status == CUDA_SUCCESS)
        return Qtrue;
    // TODO: handle status == CUDA_ERROR_INVALID_VALUE
    return Qfalse;
}

static VALUE event_record(VALUE self, VALUE rb_stream)
{
    CUevent* pevent = NULL;
    CUstream* pstream = NULL;
    CUresult status;
    Data_Get_Struct(self, CUevent, pevent);
    if (CLASS_OF(rb_stream) == rb_cCUStream) {
        Data_Get_Struct(rb_stream, CUstream, pstream);
        status = cuEventRecord(*pevent, *pstream);
    } else {
        status = cuEventRecord(*pevent, 0);
    }
    // TODO: handle status == CUDA_ERROR_INVALID_VALUE
    if (status == CUDA_ERROR_INVALID_VALUE) {}
    return self;
}

static VALUE event_synchronize(VALUE self)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    CUresult status = cuEventSynchronize(*p);
    // TODO: handle status == CUDA_ERROR_INVALID_VALUE
    if (status == CUDA_ERROR_INVALID_VALUE) {}
    return self;
}

static VALUE event_elapsed_time(VALUE klass, VALUE event_start, VALUE event_end)
{
    CUevent* pevent_start;
    CUevent* pevent_end;
    Data_Get_Struct(event_start, CUevent, pevent_start);
    Data_Get_Struct(event_end, CUevent, pevent_end);
    float etime;
    cuEventElapsedTime(&etime, *pevent_start, *pevent_end);
    return DBL2NUM(etime);
}
// }}}


// {{{ Buffer
static void memory_buffer_free(void* p)
{
    MemoryBuffer* pbuffer = static_cast<MemoryBuffer*>(p);
    delete[] pbuffer->p;
    delete pbuffer;
}

static VALUE memory_buffer_alloc(VALUE klass)
{
    MemoryBuffer* pbuffer = new MemoryBuffer;
    pbuffer->size = 0;
    pbuffer->p = NULL;
    return Data_Wrap_Struct(klass, 0, memory_buffer_free, pbuffer);
}

static VALUE memory_buffer_initialize(VALUE self, VALUE nbytes)
{
    size_t n = NUM2SIZET(nbytes);
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    pbuffer->size = n;
    pbuffer->p = new char[n];
    std::memset(static_cast<void*>(pbuffer->p), 0, pbuffer->size);
    return self;
}

static VALUE memory_buffer_size(VALUE self)
{
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    return LONG2NUM(pbuffer->size);
}

template <typename TElement>
static void buffer_free(void* p)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    TBuffer* pbuffer = static_cast<TBuffer*>(p);
    delete[] pbuffer->p;
    delete pbuffer;
}

template <typename TElement>
static VALUE buffer_alloc(VALUE klass)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    TBuffer* pbuffer = new TBuffer;
    pbuffer->size = 0;
    pbuffer->p = NULL;
    return Data_Wrap_Struct(klass, 0, &buffer_free<TElement>, pbuffer);
}

template <typename TElement>
static VALUE buffer_initialize(VALUE self, VALUE nelements)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    size_t n = NUM2SIZET(nelements);
    TBuffer* pbuffer;
    Data_Get_Struct(self, TBuffer, pbuffer);
    pbuffer->size = n*sizeof(TElement);
    pbuffer->p = reinterpret_cast<char*>(new TElement[n]);
    std::memset(static_cast<void*>(pbuffer->p), 0, pbuffer->size);
    return self;
}
typedef VALUE (*BufferInitializeFunctionType)(VALUE, VALUE);

template <typename TElement>
static VALUE buffer_element_get(VALUE self, VALUE index)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    size_t i = NUM2SIZET(index);
    TBuffer* pbuffer;
    Data_Get_Struct(self, TBuffer, pbuffer);
    TElement* e = reinterpret_cast<TElement*>(pbuffer->p);
    TElement element = e[i];
    return to_rb<TElement>(element);
}
typedef VALUE (*BufferElementGetFunctionType)(VALUE, VALUE);

template <typename TElement>
static VALUE buffer_element_set(VALUE self, VALUE index, VALUE value)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    size_t i = NUM2SIZET(index);
    TElement v = to_ctype<TElement>(value);
    TBuffer* pbuffer;
    Data_Get_Struct(self, TBuffer, pbuffer);
    TElement* e = reinterpret_cast<TElement*>(pbuffer->p);
    e[i] = v;
    return value;
}
typedef VALUE (*BufferElementSetFunctionType)(VALUE, VALUE, VALUE);
// }}}


// {{{ Memory transfer functions.
static VALUE memcpy_htod(VALUE self, VALUE rb_device_ptr, VALUE rb_memory, VALUE rb_nbytes)
{
    CUdeviceptr* pdevice_ptr;
    MemoryBuffer* pmem;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryBuffer, pmem);
    size_t nbytes = NUM2SIZET(rb_nbytes);
    cuMemcpyHtoD(*pdevice_ptr, static_cast<void*>(pmem->p), nbytes);
    return Qnil; // TODO: Return the status of the transfer.
}

static VALUE memcpy_dtoh(VALUE self, VALUE rb_memory, VALUE rb_device_ptr, VALUE rb_nbytes)
{
    MemoryBuffer* pmem;
    CUdeviceptr* pdevice_ptr;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryBuffer, pmem);
    size_t nbytes = NUM2SIZET(rb_nbytes);
    cuMemcpyDtoH(static_cast<void*>(pmem->p), *pdevice_ptr, nbytes);
    return Qnil; // TODO: Return the status of the transfer.
}
// }}}


// {{{ Driver
static VALUE driver_get_version()
{
    int v;
    cuDriverGetVersion(&v);
    return INT2FIX(v);
}
// }}}


extern "C" void Init_rubycu()
{
    rb_mSGC = rb_define_module("SGC");
    rb_mCU  = rb_define_module_under(rb_mSGC, "CU");

    rb_cCUDevice = rb_define_class_under(rb_mCU, "CUDevice", rb_cObject);
    rb_define_singleton_method(rb_cCUDevice, "get_count", (VALUE(*)(ANYARGS))device_get_count, 0);
    rb_define_alloc_func(rb_cCUDevice, device_alloc);
    rb_define_method(rb_cCUDevice, "initialize", (VALUE(*)(ANYARGS))device_initialize, -1);
    rb_define_method(rb_cCUDevice, "get"       , (VALUE(*)(ANYARGS))device_get       ,  1);
    rb_define_method(rb_cCUDevice, "get_name"  , (VALUE(*)(ANYARGS))device_get_name  ,  0);
    rb_define_method(rb_cCUDevice, "compute_capability", (VALUE(*)(ANYARGS))device_compute_capability, 0);
    rb_define_method(rb_cCUDevice, "get_attribute"     , (VALUE(*)(ANYARGS))device_get_attribute     , 1);
    rb_define_method(rb_cCUDevice, "total_mem"         , (VALUE(*)(ANYARGS))device_total_mem         , 0);

    rb_cCUComputeMode = rb_define_class_under(rb_mCU, "CUComputeMode", rb_cObject);
    rb_define_const(rb_cCUComputeMode, "DEFAULT"   , INT2FIX(CU_COMPUTEMODE_DEFAULT));
    rb_define_const(rb_cCUComputeMode, "EXCLUSIVE" , INT2FIX(CU_COMPUTEMODE_EXCLUSIVE));
    rb_define_const(rb_cCUComputeMode, "PROHIBITED", INT2FIX(CU_COMPUTEMODE_PROHIBITED));

    rb_cCUDeviceAttribute = rb_define_class_under(rb_mCU, "CUDeviceAttribute", rb_cObject);
    rb_define_const(rb_cCUDeviceAttribute, "MAX_THREADS_PER_BLOCK"            , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_BLOCK_DIM_X"                  , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_BLOCK_DIM_Y"                  , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_BLOCK_DIM_Z"                  , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_GRID_DIM_X"                   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_GRID_DIM_Y"                   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_GRID_DIM_Z"                   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_REGISTERS_PER_BLOCK"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_SHARED_MEMORY_PER_BLOCK"      , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttribute, "TOTAL_CONSTANT_MEMORY"            , INT2FIX(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY));
    rb_define_const(rb_cCUDeviceAttribute, "WARP_SIZE"                        , INT2FIX(CU_DEVICE_ATTRIBUTE_WARP_SIZE));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_PITCH"                        , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_PITCH));
    rb_define_const(rb_cCUDeviceAttribute, "CLOCK_RATE"                       , INT2FIX(CU_DEVICE_ATTRIBUTE_CLOCK_RATE));
    rb_define_const(rb_cCUDeviceAttribute, "TEXTURE_ALIGNMENT"                , INT2FIX(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT));
    rb_define_const(rb_cCUDeviceAttribute, "GPU_OVERLAP"                      , INT2FIX(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP));
    rb_define_const(rb_cCUDeviceAttribute, "MULTIPROCESSOR_COUNT"             , INT2FIX(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
    rb_define_const(rb_cCUDeviceAttribute, "KERNEL_EXEC_TIMEOUT"              , INT2FIX(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT));
    rb_define_const(rb_cCUDeviceAttribute, "INTEGRATED"                       , INT2FIX(CU_DEVICE_ATTRIBUTE_INTEGRATED));
    rb_define_const(rb_cCUDeviceAttribute, "CAN_MAP_HOST_MEMORY"              , INT2FIX(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY));
    rb_define_const(rb_cCUDeviceAttribute, "COMPUTE_MODE"                     , INT2FIX(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE1D_WIDTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_WIDTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE3D_WIDTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_HEIGHT"         , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE3D_HEIGHT"         , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE3D_DEPTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_ARRAY_WIDTH"    , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_ARRAY_HEIGHT"   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES));
    rb_define_const(rb_cCUDeviceAttribute, "SURFACE_ALIGNMENT"                , INT2FIX(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT));
    rb_define_const(rb_cCUDeviceAttribute, "CONCURRENT_KERNELS"               , INT2FIX(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS));
    rb_define_const(rb_cCUDeviceAttribute, "ECC_ENABLED"                      , INT2FIX(CU_DEVICE_ATTRIBUTE_ECC_ENABLED));
    rb_define_const(rb_cCUDeviceAttribute, "PCI_BUS_ID"                       , INT2FIX(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID));
    rb_define_const(rb_cCUDeviceAttribute, "PCI_DEVICE_ID"                    , INT2FIX(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID));

    rb_cCUContext = rb_define_class_under(rb_mCU, "CUContext", rb_cObject);
    rb_define_alloc_func(rb_cCUContext, context_alloc);
    rb_define_method(rb_cCUContext, "initialize"  , (VALUE(*)(ANYARGS))context_initialize  , -1);
    rb_define_method(rb_cCUContext, "create"      , (VALUE(*)(ANYARGS))context_create      ,  2);
    rb_define_method(rb_cCUContext, "destroy"     , (VALUE(*)(ANYARGS))context_destroy     ,  0);
    rb_define_method(rb_cCUContext, "attach"      , (VALUE(*)(ANYARGS))context_attach      , -1);
    rb_define_method(rb_cCUContext, "detach"      , (VALUE(*)(ANYARGS))context_detach      ,  0);
    rb_define_method(rb_cCUContext, "push_current", (VALUE(*)(ANYARGS))context_push_current,  0);
    rb_define_singleton_method(rb_cCUContext, "get_device" , (VALUE(*)(ANYARGS))context_get_device , 1);
    rb_define_singleton_method(rb_cCUContext, "get_limit"  , (VALUE(*)(ANYARGS))context_get_limit  , 1);
    rb_define_singleton_method(rb_cCUContext, "set_limit"  , (VALUE(*)(ANYARGS))context_set_limit  , 2);
    rb_define_singleton_method(rb_cCUContext, "pop_current", (VALUE(*)(ANYARGS))context_pop_current, 1);
    rb_define_singleton_method(rb_cCUContext, "synchronize", (VALUE(*)(ANYARGS))context_synchronize, 0);

    rb_cCUContextFlags = rb_define_class_under(rb_mCU, "CUContextFlags", rb_cObject);
    rb_define_const(rb_cCUContextFlags, "SCHED_AUTO"        , INT2FIX(CU_CTX_SCHED_AUTO));
    rb_define_const(rb_cCUContextFlags, "SCHED_SPIN"        , INT2FIX(CU_CTX_SCHED_SPIN));
    rb_define_const(rb_cCUContextFlags, "SCHED_YIELD"       , INT2FIX(CU_CTX_SCHED_YIELD));
    rb_define_const(rb_cCUContextFlags, "BLOCKING_SYNC"     , INT2FIX(CU_CTX_BLOCKING_SYNC));
    rb_define_const(rb_cCUContextFlags, "MAP_HOST"          , INT2FIX(CU_CTX_MAP_HOST));
    rb_define_const(rb_cCUContextFlags, "LMEM_RESIZE_TO_MAX", INT2FIX(CU_CTX_LMEM_RESIZE_TO_MAX));

    rb_cCULimit = rb_define_class_under(rb_mCU, "CULimit", rb_cObject);
    rb_define_const(rb_cCULimit, "STACK_SIZE"      , INT2FIX(CU_LIMIT_STACK_SIZE));
    rb_define_const(rb_cCULimit, "PRINTF_FIFO_SIZE", INT2FIX(CU_LIMIT_PRINTF_FIFO_SIZE));

    rb_cCUModule = rb_define_class_under(rb_mCU, "CUModule", rb_cObject);
    rb_define_alloc_func(rb_cCUModule, module_alloc);
    rb_define_method(rb_cCUModule, "initialize"  , (VALUE(*)(ANYARGS))module_initialize  , -1);
    rb_define_method(rb_cCUModule, "load"        , (VALUE(*)(ANYARGS))module_load        ,  1);
    rb_define_method(rb_cCUModule, "unload"      , (VALUE(*)(ANYARGS))module_unload      ,  0);
    rb_define_method(rb_cCUModule, "get_function", (VALUE(*)(ANYARGS))module_get_function,  1);
    rb_define_method(rb_cCUModule, "get_global"  , (VALUE(*)(ANYARGS))module_get_global  ,  1);

    rb_cCUDevicePtr = rb_define_class_under(rb_mCU, "CUDevicePtr", rb_cObject);
    rb_define_alloc_func(rb_cCUDevicePtr, device_ptr_alloc);
    rb_define_method(rb_cCUDevicePtr, "initialize", (VALUE(*)(ANYARGS))device_ptr_initialize, -1);
    rb_define_method(rb_cCUDevicePtr, "mem_alloc" , (VALUE(*)(ANYARGS))device_ptr_mem_alloc ,  1);
    rb_define_method(rb_cCUDevicePtr, "mem_free"  , (VALUE(*)(ANYARGS))device_ptr_mem_free  ,  0);

    rb_cCUFunction = rb_define_class_under(rb_mCU, "CUFunction", rb_cObject);
    rb_define_alloc_func(rb_cCUFunction, function_alloc);
    rb_define_method(rb_cCUFunction, "initialize"     , (VALUE(*)(ANYARGS))function_initialize     , -1);
    rb_define_method(rb_cCUFunction, "set_param"      , (VALUE(*)(ANYARGS))function_set_param      , -1);
    rb_define_method(rb_cCUFunction, "set_block_shape", (VALUE(*)(ANYARGS))function_set_block_shape, -1);
    rb_define_method(rb_cCUFunction, "set_shared_size", (VALUE(*)(ANYARGS))function_set_shared_size,  1);
    rb_define_method(rb_cCUFunction, "launch"         , (VALUE(*)(ANYARGS))function_launch         ,  0);
    rb_define_method(rb_cCUFunction, "launch_grid"    , (VALUE(*)(ANYARGS))function_launch_grid    , -1);
    rb_define_method(rb_cCUFunction, "get_attribute"  , (VALUE(*)(ANYARGS))function_get_attribute  ,  1);
    rb_define_method(rb_cCUFunction, "set_cache_config", (VALUE(*)(ANYARGS))function_set_cache_config, 1);

    rb_cCUFunctionAttribute = rb_define_class_under(rb_mCU, "CUFunctionAttribute", rb_cObject);
    rb_define_const(rb_cCUFunctionAttribute, "MAX_THREADS_PER_BLOCK", INT2FIX(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
    rb_define_const(rb_cCUFunctionAttribute, "SHARED_SIZE_BYTES"    , INT2FIX(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
    rb_define_const(rb_cCUFunctionAttribute, "CONST_SIZE_BYTES"     , INT2FIX(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES));
    rb_define_const(rb_cCUFunctionAttribute, "LOCAL_SIZE_BYTES"     , INT2FIX(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
    rb_define_const(rb_cCUFunctionAttribute, "NUM_REGS"             , INT2FIX(CU_FUNC_ATTRIBUTE_NUM_REGS));
    rb_define_const(rb_cCUFunctionAttribute, "PTX_VERSION"          , INT2FIX(CU_FUNC_ATTRIBUTE_PTX_VERSION));
    rb_define_const(rb_cCUFunctionAttribute, "BINARY_VERSION"       , INT2FIX(CU_FUNC_ATTRIBUTE_BINARY_VERSION));

    rb_cCUFunctionCache = rb_define_class_under(rb_mCU, "CUFunctionCache", rb_cObject);
    rb_define_const(rb_cCUFunctionCache, "PREFER_NONE"  , INT2FIX(CU_FUNC_CACHE_PREFER_NONE));
    rb_define_const(rb_cCUFunctionCache, "PREFER_SHARED", INT2FIX(CU_FUNC_CACHE_PREFER_SHARED));
    rb_define_const(rb_cCUFunctionCache, "PREFER_L1"    , INT2FIX(CU_FUNC_CACHE_PREFER_L1));

    rb_cCUStream = rb_define_class_under(rb_mCU, "CUStream", rb_cObject);
    rb_define_alloc_func(rb_cCUStream, stream_alloc);
    rb_define_method(rb_cCUStream, "initialize" , (VALUE(*)(ANYARGS))stream_initialize , 0);
    rb_define_method(rb_cCUStream, "create"     , (VALUE(*)(ANYARGS))stream_create     , 1);
    rb_define_method(rb_cCUStream, "destroy"    , (VALUE(*)(ANYARGS))stream_destroy    , 0);
    rb_define_method(rb_cCUStream, "query"      , (VALUE(*)(ANYARGS))stream_query      , 0);
    rb_define_method(rb_cCUStream, "synchronize", (VALUE(*)(ANYARGS))stream_synchronize, 0);

    rb_cCUEvent = rb_define_class_under(rb_mCU, "CUEvent", rb_cObject);
    rb_define_alloc_func(rb_cCUEvent, event_alloc);
    rb_define_method(rb_cCUEvent, "initialize" , (VALUE(*)(ANYARGS))event_initialize , 0);
    rb_define_method(rb_cCUEvent, "create"     , (VALUE(*)(ANYARGS))event_create     , 1);
    rb_define_method(rb_cCUEvent, "destroy"    , (VALUE(*)(ANYARGS))event_destroy    , 0);
    rb_define_method(rb_cCUEvent, "query"      , (VALUE(*)(ANYARGS))event_query      , 0);
    rb_define_method(rb_cCUEvent, "record"     , (VALUE(*)(ANYARGS))event_record     , 1);
    rb_define_method(rb_cCUEvent, "synchronize", (VALUE(*)(ANYARGS))event_synchronize, 0);
    rb_define_singleton_method(rb_cCUEvent, "elapsed_time", (VALUE(*)(ANYARGS))event_elapsed_time, 2);

    rb_cCUResult = rb_define_class_under(rb_mCU, "CUResult", rb_cObject);
    rb_define_const(rb_cCUResult, "SUCCESS"                             , INT2FIX(CUDA_SUCCESS));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_VALUE"                 , INT2FIX(CUDA_ERROR_INVALID_VALUE));
    rb_define_const(rb_cCUResult, "ERROR_OUT_OF_MEMORY"                 , INT2FIX(CUDA_ERROR_OUT_OF_MEMORY));
    rb_define_const(rb_cCUResult, "ERROR_NOT_INITIALIZED"               , INT2FIX(CUDA_ERROR_NOT_INITIALIZED));
    rb_define_const(rb_cCUResult, "ERROR_DEINITIALIZED"                 , INT2FIX(CUDA_ERROR_DEINITIALIZED));
    rb_define_const(rb_cCUResult, "ERROR_NO_DEVICE"                     , INT2FIX(CUDA_ERROR_NO_DEVICE));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_DEVICE"                , INT2FIX(CUDA_ERROR_INVALID_DEVICE));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_IMAGE"                 , INT2FIX(CUDA_ERROR_INVALID_IMAGE));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_CONTEXT"               , INT2FIX(CUDA_ERROR_INVALID_CONTEXT));
    rb_define_const(rb_cCUResult, "ERROR_CONTEXT_ALREADY_CURRENT"       , INT2FIX(CUDA_ERROR_CONTEXT_ALREADY_CURRENT));
    rb_define_const(rb_cCUResult, "ERROR_MAP_FAILED"                    , INT2FIX(CUDA_ERROR_MAP_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_UNMAP_FAILED"                  , INT2FIX(CUDA_ERROR_UNMAP_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_ARRAY_IS_MAPPED"               , INT2FIX(CUDA_ERROR_ARRAY_IS_MAPPED));
    rb_define_const(rb_cCUResult, "ERROR_ALREADY_MAPPED"                , INT2FIX(CUDA_ERROR_ALREADY_MAPPED));
    rb_define_const(rb_cCUResult, "ERROR_NO_BINARY_FOR_GPU"             , INT2FIX(CUDA_ERROR_NO_BINARY_FOR_GPU));
    rb_define_const(rb_cCUResult, "ERROR_ALREADY_ACQUIRED"              , INT2FIX(CUDA_ERROR_ALREADY_ACQUIRED));
    rb_define_const(rb_cCUResult, "ERROR_NOT_MAPPED"                    , INT2FIX(CUDA_ERROR_NOT_MAPPED));
    rb_define_const(rb_cCUResult, "ERROR_NOT_MAPPED_AS_ARRAY"           , INT2FIX(CUDA_ERROR_NOT_MAPPED_AS_ARRAY));
    rb_define_const(rb_cCUResult, "ERROR_NOT_MAPPED_AS_POINTER"         , INT2FIX(CUDA_ERROR_NOT_MAPPED_AS_POINTER));
    rb_define_const(rb_cCUResult, "ERROR_ECC_UNCORRECTABLE"             , INT2FIX(CUDA_ERROR_ECC_UNCORRECTABLE));
    rb_define_const(rb_cCUResult, "ERROR_UNSUPPORTED_LIMIT"             , INT2FIX(CUDA_ERROR_UNSUPPORTED_LIMIT));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_SOURCE"                , INT2FIX(CUDA_ERROR_INVALID_SOURCE));
    rb_define_const(rb_cCUResult, "ERROR_FILE_NOT_FOUND"                , INT2FIX(CUDA_ERROR_FILE_NOT_FOUND));
    rb_define_const(rb_cCUResult, "ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", INT2FIX(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND));
    rb_define_const(rb_cCUResult, "ERROR_SHARED_OBJECT_INIT_FAILED"     , INT2FIX(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_HANDLE"                , INT2FIX(CUDA_ERROR_INVALID_HANDLE));
    rb_define_const(rb_cCUResult, "ERROR_NOT_FOUND"                     , INT2FIX(CUDA_ERROR_NOT_FOUND));
    rb_define_const(rb_cCUResult, "ERROR_NOT_READY"                     , INT2FIX(CUDA_ERROR_NOT_READY));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_FAILED"                 , INT2FIX(CUDA_ERROR_LAUNCH_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_OUT_OF_RESOURCES"       , INT2FIX(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_TIMEOUT"                , INT2FIX(CUDA_ERROR_LAUNCH_TIMEOUT));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_INCOMPATIBLE_TEXTURING" , INT2FIX(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING));
    rb_define_const(rb_cCUResult, "ERROR_POINTER_IS_64BIT"              , INT2FIX(CUDA_ERROR_POINTER_IS_64BIT));
    rb_define_const(rb_cCUResult, "ERROR_SIZE_IS_64BIT"                 , INT2FIX(CUDA_ERROR_SIZE_IS_64BIT));
    rb_define_const(rb_cCUResult, "ERROR_UNKNOWN"                       , INT2FIX(CUDA_ERROR_UNKNOWN));

    rb_eCUStandardError = rb_define_class_under(rb_mCU, "CUStandardError", rb_eStandardError);

    rb_eCUDeviceError               = rb_define_class_under(rb_mCU, "CUDeviceError"              , rb_eCUStandardError);
    rb_eCUDeviceNotInitializedError = rb_define_class_under(rb_mCU, "CUDeviceNotInitializedError", rb_eCUDeviceError);
    rb_eCUDeviceDeinitializedError  = rb_define_class_under(rb_mCU, "CUDeviceDeinitializedError" , rb_eCUDeviceError);
    rb_eCUNoDeviceError             = rb_define_class_under(rb_mCU, "CUNoDeviceError"            , rb_eCUDeviceError);
    rb_eCUInvalidDeviceError        = rb_define_class_under(rb_mCU, "CUInvalidDeviceError"       , rb_eCUDeviceError);

    rb_eCUMapError                = rb_define_class_under(rb_mCU, "CUMapError"               , rb_eCUStandardError);
    rb_eCUMapFailedError          = rb_define_class_under(rb_mCU, "CUMapFailedError"         , rb_eCUMapError);
    rb_eCUUnMapFailedError        = rb_define_class_under(rb_mCU, "CUUnMapFailedError"       , rb_eCUMapError);
    rb_eCUArrayIsMappedError      = rb_define_class_under(rb_mCU, "CUArrayIsMappedError"     , rb_eCUMapError);
    rb_eCUAlreadyMappedError      = rb_define_class_under(rb_mCU, "CUAlreadyMappedError"     , rb_eCUMapError);
    rb_eCUNotMappedError          = rb_define_class_under(rb_mCU, "CUNotMappedError"         , rb_eCUMapError);
    rb_eCUNotMappedAsArrayError   = rb_define_class_under(rb_mCU, "CUNotMappedAsArrayError"  , rb_eCUMapError);
    rb_eCUNotMappedAsPointerError = rb_define_class_under(rb_mCU, "CUNotMappedAsPointerError", rb_eCUMapError);

    rb_eCUContextError               = rb_define_class_under(rb_mCU, "CUContextError"              , rb_eCUStandardError);
    rb_eCUInvalidContextError        = rb_define_class_under(rb_mCU, "CUInvalidContextError"       , rb_eCUContextError);
    rb_eCUContextAlreadyCurrentError = rb_define_class_under(rb_mCU, "CUContextAlreadyCurrentError", rb_eCUContextError);
    rb_eCUUnsupportedLimitError      = rb_define_class_under(rb_mCU, "CUUnsupportedLimitError"     , rb_eCUContextError);

    rb_eCULaunchError                      = rb_define_class_under(rb_mCU, "CULaunchError"                     , rb_eCUStandardError);
    rb_eCULaunchFailedError                = rb_define_class_under(rb_mCU, "CULaunchFailedError"               , rb_eCULaunchError);
    rb_eCULaunchOutOfResourcesError        = rb_define_class_under(rb_mCU, "CULaunchOutOfResourcesError"       , rb_eCULaunchError);
    rb_eCULaunchTimeoutError               = rb_define_class_under(rb_mCU, "CULaunchTimeoutError"              , rb_eCULaunchError);
    rb_eCULaunchIncompatibleTexturingError = rb_define_class_under(rb_mCU, "CULaunchIncompatibleTexturingError", rb_eCULaunchError);

    rb_eCUBitWidthError       = rb_define_class_under(rb_mCU, "CUBitWidthError"      , rb_eCUStandardError);
    rb_eCUPointerIs64BitError = rb_define_class_under(rb_mCU, "CUPointerIs64BitError", rb_eCUBitWidthError);
    rb_eCUSizeIs64BitError    = rb_define_class_under(rb_mCU, "CUSizeIs64BitError"   , rb_eCUBitWidthError);

    rb_eCUParameterError     = rb_define_class_under(rb_mCU, "CUParameterError"    , rb_eCUStandardError);
    rb_eCUInvalidValueError  = rb_define_class_under(rb_mCU, "CUInvalidValueError" , rb_eCUParameterError);
    rb_eCUInvalidHandleError = rb_define_class_under(rb_mCU, "CUInvalidHandleError", rb_eCUParameterError);

    rb_eCUMemoryError      = rb_define_class_under(rb_mCU, "CUMemoryError"     , rb_eCUStandardError);
    rb_eCUOutOfMemoryError = rb_define_class_under(rb_mCU, "CUOutOfMemoryError", rb_eCUMemoryError);

    rb_eCULibraryError                    = rb_define_class_under(rb_mCU, "CULibraryError"                   , rb_eCUStandardError);
    rb_eCUSharedObjectSymbolNotFoundError = rb_define_class_under(rb_mCU, "CUSharedObjectSymbolNotFoundError", rb_eCULibraryError);
    rb_eCUSharedObjectInitFailedError     = rb_define_class_under(rb_mCU, "CUSharedObjectInitFailedError"    , rb_eCULibraryError);

    rb_eCUHardwareError         = rb_define_class_under(rb_mCU, "CUHardwareError"        , rb_eCUStandardError);
    rb_eCUECCUncorrectableError = rb_define_class_under(rb_mCU, "CUECCUncorrectableError", rb_eCUHardwareError);

    rb_eCUFileError           = rb_define_class_under(rb_mCU, "CUFileError"          , rb_eCUStandardError);
    rb_eCUNoBinaryForGPUError = rb_define_class_under(rb_mCU, "CUNoBinaryForGPUError", rb_eCUFileError);
    rb_eCUFileNotFoundError   = rb_define_class_under(rb_mCU, "CUFileNotFoundError"  , rb_eCUFileError);
    rb_eCUInvalidSourceError  = rb_define_class_under(rb_mCU, "CUInvalidSourceError" , rb_eCUFileError);
    rb_eCUInvalidImageError   = rb_define_class_under(rb_mCU, "CUInvalidImageError"  , rb_eCUFileError);

    rb_eCUReferenceError         = rb_define_class_under(rb_mCU, "CUReferenceError"        , rb_eCUStandardError);
    rb_eCUReferenceNotFoundError = rb_define_class_under(rb_mCU, "CUReferenceNotFoundError", rb_eCUReferenceError);

    rb_eCUOtherError           = rb_define_class_under(rb_mCU, "CUOtherError"          , rb_eCUStandardError);
    rb_eCUAlreadyAcquiredError = rb_define_class_under(rb_mCU, "CUAlreadyAcquiredError", rb_eCUOtherError);
    rb_eCUNotReadyError        = rb_define_class_under(rb_mCU, "CUNotReadyError"       , rb_eCUOtherError);

    rb_eCUUnknownError = rb_define_class_under(rb_mCU, "CUUnknownError", rb_eCUStandardError);

    rb_error_class_by_enum = rb_hash_new();
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NOT_INITIALIZED), rb_eCUDeviceNotInitializedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_DEINITIALIZED)  , rb_eCUDeviceDeinitializedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NO_DEVICE)      , rb_eCUNoDeviceError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_INVALID_DEVICE) , rb_eCUInvalidDeviceError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_MAP_FAILED)           , rb_eCUMapFailedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_UNMAP_FAILED)         , rb_eCUUnMapFailedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_ARRAY_IS_MAPPED)      , rb_eCUArrayIsMappedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_ALREADY_MAPPED)       , rb_eCUAlreadyMappedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NOT_MAPPED)           , rb_eCUNotMappedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NOT_MAPPED_AS_ARRAY)  , rb_eCUNotMappedAsArrayError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NOT_MAPPED_AS_POINTER), rb_eCUNotMappedAsPointerError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_INVALID_CONTEXT)        , rb_eCUInvalidContextError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_CONTEXT_ALREADY_CURRENT), rb_eCUContextAlreadyCurrentError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_UNSUPPORTED_LIMIT)      , rb_eCUUnsupportedLimitError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_LAUNCH_FAILED)                , rb_eCULaunchFailedError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)      , rb_eCULaunchOutOfResourcesError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_LAUNCH_TIMEOUT)               , rb_eCULaunchTimeoutError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING), rb_eCULaunchIncompatibleTexturingError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_POINTER_IS_64BIT), rb_eCUPointerIs64BitError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_SIZE_IS_64BIT)   , rb_eCUSizeIs64BitError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_INVALID_VALUE)  , rb_eCUInvalidValueError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_INVALID_HANDLE) , rb_eCUInvalidHandleError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_OUT_OF_MEMORY), rb_eCUOutOfMemoryError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND), rb_eCUSharedObjectSymbolNotFoundError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)     , rb_eCUSharedObjectInitFailedError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_ECC_UNCORRECTABLE), rb_eCUECCUncorrectableError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NO_BINARY_FOR_GPU), rb_eCUNoBinaryForGPUError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_FILE_NOT_FOUND)   , rb_eCUFileNotFoundError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_INVALID_SOURCE)   , rb_eCUInvalidSourceError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_INVALID_IMAGE)    , rb_eCUInvalidImageError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NOT_FOUND), rb_eCUReferenceNotFoundError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_ALREADY_ACQUIRED), rb_eCUAlreadyAcquiredError);
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_NOT_READY)       , rb_eCUNotReadyError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_UNKNOWN), rb_eCUUnknownError);

    rb_cMemoryBuffer = rb_define_class_under(rb_mCU, "MemoryBuffer", rb_cObject);
    rb_define_alloc_func(rb_cMemoryBuffer, memory_buffer_alloc);
    rb_define_method(rb_cMemoryBuffer, "initialize", (VALUE(*)(ANYARGS))memory_buffer_initialize, 1);
    rb_define_method(rb_cMemoryBuffer, "size"      , (VALUE(*)(ANYARGS))memory_buffer_size      , 0);

    rb_cInt32Buffer = rb_define_class_under(rb_mCU, "Int32Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cInt32Buffer, buffer_alloc<int>);
    rb_define_const(rb_cInt32Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(int)));
    rb_define_method(rb_cInt32Buffer, "initialize", (VALUE(*)(ANYARGS))static_cast<BufferInitializeFunctionType>(&buffer_initialize<int>) , 1);
    rb_define_method(rb_cInt32Buffer, "[]"        , (VALUE(*)(ANYARGS))static_cast<BufferElementGetFunctionType>(&buffer_element_get<int>), 1);
    rb_define_method(rb_cInt32Buffer, "[]="       , (VALUE(*)(ANYARGS))static_cast<BufferElementSetFunctionType>(&buffer_element_set<int>), 2);

    rb_cInt64Buffer = rb_define_class_under(rb_mCU, "Int64Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cInt64Buffer, buffer_alloc<long>);
    rb_define_const(rb_cInt64Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(long)));
    rb_define_method(rb_cInt64Buffer, "initialize", (VALUE(*)(ANYARGS))static_cast<BufferInitializeFunctionType>(&buffer_initialize<long>) , 1);
    rb_define_method(rb_cInt64Buffer, "[]"        , (VALUE(*)(ANYARGS))static_cast<BufferElementGetFunctionType>(&buffer_element_get<long>), 1);
    rb_define_method(rb_cInt64Buffer, "[]="       , (VALUE(*)(ANYARGS))static_cast<BufferElementSetFunctionType>(&buffer_element_set<long>), 2);

    rb_cFloat32Buffer = rb_define_class_under(rb_mCU, "Float32Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cFloat32Buffer, buffer_alloc<float>);
    rb_define_const(rb_cFloat32Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(float)));
    rb_define_method(rb_cFloat32Buffer, "initialize", (VALUE(*)(ANYARGS))static_cast<BufferInitializeFunctionType>(&buffer_initialize<float>) , 1);
    rb_define_method(rb_cFloat32Buffer, "[]"        , (VALUE(*)(ANYARGS))static_cast<BufferElementGetFunctionType>(&buffer_element_get<float>), 1);
    rb_define_method(rb_cFloat32Buffer, "[]="       , (VALUE(*)(ANYARGS))static_cast<BufferElementSetFunctionType>(&buffer_element_set<float>), 2);

    rb_cFloat64Buffer = rb_define_class_under(rb_mCU, "Float64Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cFloat64Buffer, buffer_alloc<double>);
    rb_define_const(rb_cFloat64Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(double)));
    rb_define_method(rb_cFloat64Buffer, "initialize", (VALUE(*)(ANYARGS))static_cast<BufferInitializeFunctionType>(&buffer_initialize<double>) , 1);
    rb_define_method(rb_cFloat64Buffer, "[]"        , (VALUE(*)(ANYARGS))static_cast<BufferElementGetFunctionType>(&buffer_element_get<double>), 1);
    rb_define_method(rb_cFloat64Buffer, "[]="       , (VALUE(*)(ANYARGS))static_cast<BufferElementSetFunctionType>(&buffer_element_set<double>), 2);

    rb_define_module_function(rb_mCU, "memcpy_htod", (VALUE(*)(ANYARGS))memcpy_htod, 3);
    rb_define_module_function(rb_mCU, "memcpy_dtoh", (VALUE(*)(ANYARGS))memcpy_dtoh, 3);

    rb_define_module_function(rb_mCU, "driver_get_version", (VALUE(*)(ANYARGS))driver_get_version, 0);

    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to initialize the CUDA driver API.");
    }
}
