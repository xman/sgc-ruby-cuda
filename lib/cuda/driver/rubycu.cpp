/*
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
*/

#include <cstring>
#include "ruby.h"
#include "cuda.h"

namespace SGC {
namespace CU {

// {{{ SGC Ruby modules.
static VALUE rb_mSGC;
static VALUE rb_mCU;
static VALUE rb_mIBuffer;
static VALUE rb_mIBufferClassMethods;
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
static VALUE rb_cCUEventFlags;
static VALUE rb_cCUAddressMode;
static VALUE rb_cCUFilterMode;
static VALUE rb_cCUTexRefFlags;
static VALUE rb_cCUTexRef;
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
static VALUE rb_eCUOperatingSystemError;

static VALUE rb_eCUUnknownError;

static VALUE rb_cMemoryPointer;
static VALUE rb_cMemoryBuffer;
static VALUE rb_cInt32Buffer;
static VALUE rb_cInt64Buffer;
static VALUE rb_cFloat32Buffer;
static VALUE rb_cFloat64Buffer;
// }}}

// {{{ SGC C/C++ structures.
typedef struct {
    char* p;
} MemoryPointer;

typedef struct : MemoryPointer {
    size_t size;
    bool is_page_locked;
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

VALUE to_rb(bool b)
{
    if (b) {
        return Qtrue;
    }
    return Qfalse;
}

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
bool to_ctype<bool>(VALUE b)
{
    if (b == Qfalse || b == Qnil) {
        return false;
    }
    return true;
}

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
    VALUE v = rb_const_get(rb_class_const, SYM2ID(current_label));
    if (FIX2INT(v) == FIX2INT(constant_value)) {
        label = current_label;
        return Qtrue;
    }
    return Qfalse;
}

// Extend _klass_ with the module _mod::ClassMethods_.
static VALUE module_included_classmethods_hook(VALUE mod, VALUE klass)
{
    VALUE m = rb_cvar_get(mod, rb_intern("ClassMethods"));
    rb_extend_object(klass, m);
    return Qnil;
}

#define RAISE_CU_STD_ERROR_FORMATTED(status, format, ...) rb_raise(rb_hash_aref(rb_error_class_by_enum, INT2FIX(status)), "%s:%d " format, __FILE__, __LINE__, __VA_ARGS__)
#define RAISE_CU_STD_ERROR(status, message) RAISE_CU_STD_ERROR_FORMATTED(status, "%s", message)
// }}}

// {{{ SGC Ruby data.
static VALUE rb_error_class_by_enum;
// }}}


// {{{ CUdevice

/*  call-seq: CUDevice.get_count    ->    Fixnum
 *
 *  Return the number of CUDA devices.
 */
static VALUE device_get_count(VALUE klass)
{
    int count;
    CUresult status = cuDeviceGetCount(&count);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get device count.");
    }
    return INT2FIX(count);
}

/*  call-seq: CUDevice.get(index)    ->    CUDevice
 *
 *  Return a CUDevice instance corresponding to CUDA device _index_ (0..CUDevice.get_count-1).
 */
static VALUE device_get(VALUE klass, VALUE num)
{
    CUdevice* pdev;
    VALUE rb_pdev = rb_class_new_instance(0, NULL, rb_cCUDevice);
    Data_Get_Struct(rb_pdev, CUdevice, pdev);
    int i = FIX2INT(num);
    CUresult status = cuDeviceGet(pdev, i);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get device %d.", i);
    }
    return rb_pdev;
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

/*  call-seq: dev.get_name    ->    String
 *
 *  Return the name of _self_ with a maximum of 255 characters.
 */
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

/*  call-seq: dev.compute_capability    ->    Hash { major:, minor: }
 *
 *  Return the compute capability of _self_.
 *
 *      # For a device with compute capability 1.3:
 *      dev.compute_capability        #=> { major: 1, minor: 3 }
 */
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

/*  call-seq: dev.get_attribute(attribute)    ->    Fixnum
 *
 *  Return _attribute_ (CUDeviceAttribute) of _self_.
 *
 *      dev.get_attribute(CUDeviceAttribute::MAX_THREADS_PER_BLOCK)        #=> 512
 *      dev.get_attribute(CUDeviceAttribute::MULTIPROCESSOR_COUNT)         #=> 30
 *      dev.get_attribute(CUDeviceAttribute::MAX_SHARED_MEMORY_PER_BLOCK)  #=> 16384
 */
static VALUE device_get_attribute(VALUE self, VALUE attribute)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int v;
    CUresult status = cuDeviceGetAttribute(&v, static_cast<CUdevice_attribute>(FIX2INT(attribute)), *p);
    if (status != CUDA_SUCCESS) {
        VALUE attributes = rb_funcall(rb_cCUDeviceAttribute, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCUDeviceAttribute, attribute, Qnil };
        rb_block_call(attributes, rb_intern("find"), 0, NULL, RUBY_METHOD_FUNC(class_const_match), (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to query device attribute: %s.", rb_id2name(SYM2ID(ary[2])));
    }
    return INT2FIX(v);
}

/*  call-seq: dev.get_properties    ->    Hash
 *
 *  Return the properties of _self_ in a hash with the following keys:
 *  * :clock_rate
 *  * :max_grid_size
 *  * :max_threads_dim
 *  * :max_threads_per_block
 *  * :mem_pitch
 *  * :regs_per_block
 *  * :shared_mem_per_block
 *  * :simd_width
 *  * :texture_align
 *  * :total_constant_memory
 */
static VALUE device_get_properties(VALUE self)
{
    CUdevice* pdevice;
    Data_Get_Struct(self, CUdevice, pdevice);
    CUdevprop prop;
    CUresult status = cuDeviceGetProperties(&prop, *pdevice);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get device properties.");
    }

    VALUE max_grid_size = rb_ary_new3(3, INT2FIX(prop.maxGridSize[0]), INT2FIX(prop.maxGridSize[1]), INT2FIX(prop.maxGridSize[2]));
    VALUE max_threads_dim = rb_ary_new3(3, INT2FIX(prop.maxThreadsDim[0]), INT2FIX(prop.maxThreadsDim[1]), INT2FIX(prop.maxThreadsDim[2]));

    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("clock_rate")), INT2FIX(prop.clockRate));
    rb_hash_aset(h, ID2SYM(rb_intern("max_grid_size")), max_grid_size);
    rb_hash_aset(h, ID2SYM(rb_intern("max_threads_dim")), max_threads_dim);
    rb_hash_aset(h, ID2SYM(rb_intern("max_threads_per_block")), INT2FIX(prop.maxThreadsPerBlock));
    rb_hash_aset(h, ID2SYM(rb_intern("mem_pitch")), INT2FIX(prop.memPitch));
    rb_hash_aset(h, ID2SYM(rb_intern("regs_per_block")), INT2FIX(prop.regsPerBlock));
    rb_hash_aset(h, ID2SYM(rb_intern("shared_mem_per_block")), INT2FIX(prop.sharedMemPerBlock));
    rb_hash_aset(h, ID2SYM(rb_intern("simd_width")), INT2FIX(prop.SIMDWidth));
    rb_hash_aset(h, ID2SYM(rb_intern("texture_align")), INT2FIX(prop.textureAlign));
    rb_hash_aset(h, ID2SYM(rb_intern("total_constant_memory")), INT2FIX(prop.totalConstantMemory));
    return h;
}

/*  call-seq: dev.total_mem    ->    Numeric
 *
 *  Return the total amount of device memory in bytes.
 */
static VALUE device_total_mem(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    size_t nbytes;
    CUresult status = cuDeviceTotalMem(&nbytes, *p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get device total amount of memory available.");
    }
    return SIZET2NUM(nbytes);
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

/*  call-seq: ctx.create(device)           ->    self
 *            ctx.create(flags, device)    ->    self
 *
 *  Create a new CUDA context with _flags_ (CUContextFlags) and _device_,
 *  then associate it with the calling thread, and return the context.
 *  Setting flags to 0 or ommitting flags uses SCHED_AUTO.
 *
 *      dev = CUDevice.get(0)
 *      ctx = CUContext.new
 *      ctx.create(dev)           #=>    ctx
 *      ctx.create(0, dev)        #=>    ctx
 *      ctx.create(CUContextFlags::SCHED_SPIN | CUContextFlags::BLOCKING_SYNC, dev)        #=>    ctx
 */
static VALUE context_create(int argc, VALUE* argv, VALUE self)
{
    if (argc <= 0 || argc > 2) {
        rb_raise(rb_eArgError, "wrong number of arguments (%d for 1 or 2).", argc);
    }

    CUcontext* pcontext;
    CUdevice* pdevice;
    unsigned int flags = 0;
    Data_Get_Struct(self, CUcontext, pcontext);
    if (argc == 2) {
        flags = FIX2UINT(argv[0]);
        Data_Get_Struct(argv[1], CUdevice, pdevice);
    } else { // argc == 1
        Data_Get_Struct(argv[0], CUdevice, pdevice);
    }
    CUresult status = cuCtxCreate(pcontext, flags, *pdevice);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to create context: flags = 0x%x.", flags);
    }
    return self;
}

/*  call-seq: ctx.destroy    ->    nil
 *
 *  Destroy the CUDA context _self_.
 */
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

/*  call-seq: ctx.attach           ->    self
 *            ctx.attach(flags)    ->    self
 *
 *  Increment the reference count on _self_.
 *  Currently, _flags_ must be set to 0.
 */
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
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to attach context: flags = 0x%x.", flags);
    }
    return self;
}


/*  call-seq: ctx.detach    ->    nil
 *
 *  Decrement the reference count on _self_.
 */
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

/*  call-seq: ctx.push_current    ->    self
 *
 *  Push _self_ onto the context stack, which becomes currently active context.
 */
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

/*  call-seq: ctx.get_api_version    ->     Numeric
 *
 *  Return the API version used to create _self_.
 */
static VALUE context_get_api_version(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    unsigned int version;
    CUresult status = cuCtxGetApiVersion(*p, &version);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get the API version of this context.");
    }
    return UINT2NUM(version);
}

/*  call-seq: CUContext.get_api_version    ->    Numeric
 *
 *  Return the API version used to create current context.
 */
static VALUE context_get_api_version_singleton(VALUE klass)
{
    unsigned int version;
    CUresult status = cuCtxGetApiVersion(NULL, &version);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get the API version of current context.");
    }
    return UINT2NUM(version);
}

/*  call-seq: CUContext.get_device    ->    CUDevice
 *
 *  Return the device associated to the current CUDA context.
 */
static VALUE context_get_device(VALUE klass)
{
    VALUE device = rb_class_new_instance(0, NULL, rb_cCUDevice);
    CUdevice* pdevice;
    Data_Get_Struct(device, CUdevice, pdevice);
    CUresult status = cuCtxGetDevice(pdevice);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get current context's device.");
    }
    return device;
}

/*  call-seq: CUContext.get_limit(limit)    ->    Numeric
 *
 *  Return the _limit_ (CULimit) of the current CUDA context.
 *
 *      CUContext.get_limit(CULimit::STACK_SIZE)        #=>    8192
 */
static VALUE context_get_limit(VALUE klass, VALUE limit)
{
    CUlimit l = static_cast<CUlimit>(FIX2UINT(limit));
    size_t v = 0;
    CUresult status = cuCtxGetLimit(&v, l);
    if (status != CUDA_SUCCESS) {
        VALUE limits = rb_funcall(rb_cCULimit, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCULimit, limit, Qnil };
        rb_block_call(limits, rb_intern("find"), 0, NULL, RUBY_METHOD_FUNC(class_const_match), (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get context limit: %s.", rb_id2name(SYM2ID(ary[2])));
    }
    return SIZET2NUM(v);
}

/*  call-seq: CUContext.set_limit(limit, value)    ->    nil
 *
 *  Set the _limit_ (CULimit) of the current CUDA context.
 *
 *      CUContext.set_limit(CULimit::STACK_SIZE, 8192)        #=>    nil
 */
static VALUE context_set_limit(VALUE klass, VALUE limit, VALUE value)
{
    CUlimit l = static_cast<CUlimit>(FIX2UINT(limit));
    CUresult status = cuCtxSetLimit(l, NUM2SIZET(value));
    if (status != CUDA_SUCCESS) {
        VALUE limits = rb_funcall(rb_cCULimit, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCULimit, limit, Qnil };
        rb_block_call(limits, rb_intern("find"), 0, NULL, RUBY_METHOD_FUNC(class_const_match), (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set context limit: %s to %lu.", rb_id2name(SYM2ID(ary[2])), NUM2SIZET(value));
    }
    return Qnil;
}

/*  call-seq: CUContext.get_cache_config        ->    CUFunctionCache
 *
 *  Return the cache config of the current CUDA context.
 *
 *      CUContext.get_cache_config        #=> 1
 */
static VALUE context_get_cache_config(VALUE klass)
{
    CUfunc_cache config;
    CUresult status = cuCtxGetCacheConfig(&config);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get context cache config.");
    }
    return UINT2NUM(static_cast<unsigned int>(config));
}

/*  call-seq: CUContext.set_cache_config(config)        ->    nil
 *
 *  Set the cache with _config_ (CUFunctionCache) for the current CUDA context.
 *
 *      CUContext.set_cache_config(CUFunctionCache::PREFER_SHARED)        #=> nil
 */
static VALUE context_set_cache_config(VALUE klass, VALUE config)
{
    CUresult status = cuCtxSetCacheConfig(static_cast<CUfunc_cache>(FIX2UINT(config)));
    if (status != CUDA_SUCCESS) {
        VALUE configs = rb_funcall(rb_cCUFunctionCache, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCUFunctionCache, config, Qnil };
        rb_block_call(configs, rb_intern("find"), 0, NULL, RUBY_METHOD_FUNC(class_const_match), (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set context cache config: %s.", rb_id2name(SYM2ID(ary[2])));
    }
    return Qnil;
}

/*  call-seq: CUContext.pop_current        ->    CUContext
 *
 *  Pop the current CUDA context from the context stack, which becomes inactive.
 */
static VALUE context_pop_current(VALUE klass)
{
    VALUE context = rb_class_new_instance(0, NULL, rb_cCUContext);
    CUcontext* pcontext;
    Data_Get_Struct(context, CUcontext, pcontext);
    CUresult status = cuCtxPopCurrent(pcontext);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to pop current context.");
    }
    return context;
}

/*  call-seq: CUContext.synchronize        ->    nil
 *
 *  Block until all the tasks of the current CUDA context complete.
 */
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

/*  call-seq: mod.load(path)    ->    self
 *
 *  Load a compute module from the file at _path_ into the current CUDA context.
 *  The file should be a cubin file or a PTX file.
 *
 *  A PTX file may be obtained by compiling the .cu file using nvcc with -ptx option.
 *      $ nvcc -ptx vadd.cu
 */
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

/*  call-seq: mod.load_data(image_str)    ->    self
 *
 *  Load a compute module from the String _image_str_ which contains a cubin or a PTX data
 *  into the current CUDA context.
 *
 *  <br /> See also CUModule#load.
 */
static VALUE module_load_data(VALUE self, VALUE image)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    CUresult status = cuModuleLoadData(p, StringValuePtr(image));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to load module data.");
    }
    return self;
}

/*  call-seq: mod.unload    ->    self
 *
 *  Unload _self_ from the current CUDA context.
 */
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

/*  call-seq: mod.get_function(name_str)    ->    CUFunction
 *
 *  Return a CUFunction instance corresponding to the function name _name_str_ in the loaded compute module.
 *  A compute module was loaded with CUModule#load and alike methods.
 */
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

/*  call-seq: mod.get_global(name_str)    ->    [CUDevicePtr, Numeric]
 *
 *  Return the CUDevicePtr corresponding to the global variable in the loaded compute module and its size in bytes.
 */
static VALUE module_get_global(VALUE self, VALUE str)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    VALUE rb_devptr = device_ptr_alloc(rb_cCUDevicePtr);
    device_ptr_initialize(0, NULL, rb_devptr);
    CUdeviceptr* pdevptr;
    Data_Get_Struct(rb_devptr, CUdeviceptr, pdevptr);
    size_t nbytes;
    CUresult status = cuModuleGetGlobal(pdevptr, &nbytes, *p, StringValuePtr(str));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get module global: %s.", StringValuePtr(str));
    }
    return rb_ary_new3(2, rb_devptr, SIZET2NUM(nbytes));
}

/*  call-seq: mod.get_texref(name_str)    ->    CUTexRef
 *
 *  Return a CUTexRef instance corresponding to the texture name _name_str_ in the loaded compute module.
 */
static VALUE module_get_texref(VALUE self, VALUE str)
{
    CUmodule* pmodule;
    CUtexref* ptexref;
    Data_Get_Struct(self, CUmodule, pmodule);
    VALUE rb_texref = rb_class_new_instance(0, NULL, rb_cCUTexRef);
    Data_Get_Struct(rb_texref, CUtexref, ptexref);
    CUresult status = cuModuleGetTexRef(ptexref, *pmodule, StringValuePtr(str));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get module texture reference: %s.", StringValuePtr(str));
    }
    return rb_texref;
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

/*  call-seq: devptr.offset(offset)    ->    CUDevicePtr
 *
 *  Return a CUDevicePtr instance pointing to the memory location _offset_ (bytes) from _self_.
 */
static VALUE device_ptr_offset(VALUE self, VALUE offset)
{
    CUdeviceptr* pdevptr;
    CUdeviceptr* pdevptr_offset;
    Data_Get_Struct(self, CUdeviceptr, pdevptr);
    VALUE rb_pdevptr_offset = rb_class_new_instance(0, NULL, rb_cCUDevicePtr);
    Data_Get_Struct(rb_pdevptr_offset, CUdeviceptr, pdevptr_offset);
    *pdevptr_offset = *pdevptr + NUM2UINT(offset);
    return rb_pdevptr_offset;
}

/*  call-seq: devptr.mem_alloc(nbytes)    ->    self
 *
 *  Allocate _nbytes_ device memory and let _self_ points to this allocated memory.
 */
static VALUE device_ptr_mem_alloc(VALUE self, VALUE nbytes)
{
    CUdeviceptr* p;
    Data_Get_Struct(self, CUdeviceptr, p);
    CUresult status = cuMemAlloc(p, NUM2UINT(nbytes));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to allocate memory: size = %u.", NUM2UINT(nbytes));
    }
    return self;
}

/*  call-seq: devptr.mem_free    ->    self
 *
 *  Free the allocated device memory _self_ pointing to.
 */
static VALUE device_ptr_mem_free(VALUE self)
{
    CUdeviceptr* p;
    Data_Get_Struct(self, CUdeviceptr, p);
    CUresult status = cuMemFree(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to free memory.");
    }
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

/*  call-seq: func.set_param(arg1, arg2, *other_args)    ->    self
 *
 *  Set the argument list of _self_ to _arg1_, _arg2_, *other_args.
 */
static VALUE function_set_param(int argc, VALUE* argv, VALUE self)
{
    #define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

    int offset = 0;
    CUfunction* pfunc;
    Data_Get_Struct(self, CUfunction, pfunc);

    CUresult status = CUDA_ERROR_UNKNOWN;
    for (int i = 0; i < argc; ++i) {
        if (CLASS_OF(argv[i]) == rb_cCUDevicePtr) {
            CUdeviceptr* p;
            Data_Get_Struct(argv[i], CUdeviceptr, p);
            ALIGN_UP(offset, __alignof(*p));
            status = cuParamSetv(*pfunc, offset, p, sizeof(*p));
            if (status != CUDA_SUCCESS) break;
            offset += sizeof(*p);
        } else if (CLASS_OF(argv[i]) == rb_cFixnum) {
            int num = FIX2INT(argv[i]);
            ALIGN_UP(offset, __alignof(num));
            status = cuParamSeti(*pfunc, offset, num);
            if (status != CUDA_SUCCESS) break;
            offset += sizeof(int);
        } else if (CLASS_OF(argv[i]) == rb_cFloat) {
            float num = static_cast<float>(NUM2DBL(argv[i]));
            ALIGN_UP(offset, __alignof(num));
            status = cuParamSetf(*pfunc, offset, num);
            if (status != CUDA_SUCCESS) break;
            offset += sizeof(float);
        } else {
            rb_raise(rb_eArgError, "Invalid type of argument %d.", i+1);
        }
    }
    if (argc > 0 && status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to set function parameters.");
    }

    status = cuParamSetSize(*pfunc, offset);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to set function parameter size.");
    }
    return self;
}

/*  call-seq: func.set_texref(texref)    ->    self
 *
 *  Add the _texref_ to the argument list of _self_.
 *
 *  Note: This method is *deprecated*. This is no longer necessary.
 */
static VALUE function_set_texref(VALUE self, VALUE texref)
{
    rb_warn("CUFunction#set_texref is deprecated.");
    CUfunction* pfunc;
    CUtexref* ptexref;
    Data_Get_Struct(self, CUfunction, pfunc);
    Data_Get_Struct(texref, CUtexref, ptexref);
    CUresult status = cuParamSetTexRef(*pfunc, CU_PARAM_TR_DEFAULT, *ptexref);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to set function texture reference.");
    }
    return self;
}

/*  call-seq: func.set_block_shape(xdim)                ->    self
 *            func.set_block_shape(xdim, ydim)          ->    self
 *            func.set_block_shape(xdim, ydim, zdim)    ->    self
 *
 *  Set the block dimensions to use for next launch. _ydim_ and _zdim_ which may be omitted are default to 1.
 */
static VALUE function_set_block_shape(int argc, VALUE* argv, VALUE self)
{
    if (argc <= 0 || argc > 3) {
        rb_raise(rb_eArgError, "wrong number of arguments (%d for 1 to 3 integers).", argc);
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

    CUresult status = cuFuncSetBlockShape(*pfunc, xdim, ydim, zdim);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set function block shape: (x,y,z) = (%d,%d,%d).", xdim, ydim, zdim);
    }
    return self;
}

/*  call-seq: func.set_shared_size(nbytes)    ->    self
 *
 *  Set the dynamic shared-memory size to use for next launch.
 */
static VALUE function_set_shared_size(VALUE self, VALUE nbytes)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    CUresult status = cuFuncSetSharedSize(*p, NUM2UINT(nbytes));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set function shared memory size: %u.", NUM2UINT(nbytes));
    }
    return self;
}

/*  call-seq: func.launch    ->    self
 *
 *  Launch _self_ to execute on a CUDA device.
 */
static VALUE function_launch(VALUE self)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    CUresult status = cuLaunch(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to launch kernel function on 1x1x1 grid of blocks.");
    }
    return self;
}

/*  call-seq: func.launch_grid(xdim)          ->    self
 *            func.launch_grid(xdim, ydim)    ->    self
 *
 *  Launch _self_ with grid dimensions (xdim, ydim) to execute on a CUDA device.
 *  _ydim_ which may be omitted is default to 1.
 */
static VALUE function_launch_grid(int argc, VALUE* argv, VALUE self)
{
    if (argc <= 0 || argc > 2) {
        rb_raise(rb_eArgError, "wrong number of arguments (%d for 1 or 2 integers).", argc);
    }

    CUfunction* pfunc;
    Data_Get_Struct(self, CUfunction, pfunc);

    int xdim = FIX2INT(argv[0]);
    int ydim = 1;

    if (argc >= 2) {
        ydim = FIX2INT(argv[1]);
    }

    CUresult status = cuLaunchGrid(*pfunc, xdim, ydim);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to launch kernel function on %dx%d grid of blocks.", xdim, ydim);
    }
    return self;
}

/*  call-seq: func.launch_grid_async(xdim, stream)          ->    self
 *            func.launch_grid_async(xdim, ydim, stream)    ->    self
 *
 *  Launch _self_ with grid dimensions (xdim, ydim) on _stream_ asynchronously to execute on a CUDA device.
 *  _ydim_ which may be omitted is default to 1. Setting _stream_ to anything other than an instance of CUStream
 *  will execute on the default stream 0.
 */
static VALUE function_launch_grid_async(int argc, VALUE* argv, VALUE self)
{
    if (argc < 2 || argc > 3) {
        rb_raise(rb_eArgError, "wrong number of arguments (%d for 2 or 3).", argc);
    }

    CUfunction* pfunc;
    CUstream *pstream = NULL;
    CUstream stream0 = 0;
    Data_Get_Struct(self, CUfunction, pfunc);

    int xdim = FIX2INT(argv[0]);
    int ydim = 1;

    if (argc == 2) {
        if (CLASS_OF(argv[1]) == rb_cCUStream) {
            Data_Get_Struct(argv[1], CUstream, pstream);
        } else {
            pstream = &stream0;
        }
    } else if (argc == 3) {
        ydim = FIX2INT(argv[1]);
        if (CLASS_OF(argv[2]) == rb_cCUStream) {
            Data_Get_Struct(argv[2], CUstream, pstream);
        } else {
            pstream = &stream0;
        }
    }

    CUresult status = cuLaunchGridAsync(*pfunc, xdim, ydim, *pstream);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to launch kernel function asynchronously on %dx%d grid of blocks.", xdim, ydim);
    }
    return self;
}

/*  call-seq: func.get_attribute(attribute)    ->    Fixnum
 *
 *  Return _attribute_ (CUFunctionAttribute) of _self_.
 *
 *      func.get_attribute(CUFunctionAttribute::MAX_THREADS_PER_BLOCK)    #=> 512
 *      func.get_attribute(CUFunctionAttribute::SHARED_SIZE_BYTES)        #=> 44
 *      func.get_attribute(CUFunctionAttribute::NUM_REGS)                 #=> 3
 */
static VALUE function_get_attribute(VALUE self, VALUE attribute)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    int v;
    CUresult status = cuFuncGetAttribute(&v, static_cast<CUfunction_attribute>(FIX2INT(attribute)), *p);
    if (status != CUDA_SUCCESS) {
        VALUE attributes = rb_funcall(rb_cCUFunctionAttribute, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCUFunctionAttribute, attribute, Qnil };
        rb_block_call(attributes, rb_intern("find"), 0, NULL, RUBY_METHOD_FUNC(class_const_match), (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to query function attribute: %s.", rb_id2name(SYM2ID(ary[2])));
    }
    return INT2FIX(v);
}

/*  call-seq: func.set_cache_config(config)    ->    self
 *
 *  Set the preferred cache configuration (CUFunctionCache) to use for next launch.
 */
static VALUE function_set_cache_config(VALUE self, VALUE config)
{
    CUfunction* p;
    Data_Get_Struct(self, CUfunction, p);
    CUresult status = cuFuncSetCacheConfig(*p, static_cast<CUfunc_cache>(FIX2UINT(config)));
    if (status != CUDA_SUCCESS) {
        VALUE configs = rb_funcall(rb_cCUFunctionCache, rb_intern("constants"), 0);
        VALUE ary[3] = { rb_cCUFunctionCache, config, Qnil };
        rb_block_call(configs, rb_intern("find"), 0, NULL, RUBY_METHOD_FUNC(class_const_match), (VALUE)ary);
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set function cache config: %s.", rb_id2name(SYM2ID(ary[2])));
    }
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

/*  call-seq: stream.create(flags)    ->    self
 *
 *  Create a stream and set _self_ to this stream. Currently, _flags_ must be set to 0.
 */
static VALUE stream_create(VALUE self, VALUE flags)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    CUresult status = cuStreamCreate(p, FIX2UINT(flags));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to create stream: flags = 0x%x", FIX2UINT(flags));
    }
    return self;
}

/*  call-seq: stream.destroy    ->    nil
 *
 *  Destroy the stream _self_.
 */
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

/*  call-seq: stream.query    ->    true or false
 *
 *  Return true if all operations in _self_ have completed. Otherwise, return false.
 */
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

/*  call-seq: stream.synchronize    ->    self
 *
 *  Block until all operations in _self_ complete.
 */
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

/*  call-seq: event.create(flags)    ->    self
 *
 *  Create an event with _flags_ (CUEventFlags) and set _self_ to this event.
 *
 *      event.create(CUEventFlags::DEFAULT)              #=> self
 *      event.create(CUEventFlags::BLOCKING_SYNC)        #=> self
 */
static VALUE event_create(VALUE self, VALUE flags)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    CUresult status = cuEventCreate(p, FIX2UINT(flags));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to create event: flags = 0x%x.", FIX2UINT(flags));
    }
    return self;
}

/*  call-seq: event.destroy    ->    nil
 *
 *  Destroy the event _self_.
 */
static VALUE event_destroy(VALUE self)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    CUresult status = cuEventDestroy(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to destroy event.");
    }
    return Qnil;
}

/*  call-seq: event.query    ->    true or false
 *
 *  Return true if _self_ has been recorded. Otherwise, return false.
 */
static VALUE event_query(VALUE self)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    CUresult status = cuEventQuery(*p);
    if (status == CUDA_SUCCESS) {
        return Qtrue;
    } else if (status == CUDA_ERROR_NOT_READY) {
        return Qfalse;
    } else if (status == CUDA_ERROR_INVALID_VALUE) {
        RAISE_CU_STD_ERROR(status, "Failed to query event: cuEventRecord() has not been called on this event.");
    } else {
        RAISE_CU_STD_ERROR(status, "Failed to query event.");
    }
}

/*  call-seq: event.record(stream)    ->    self
 *
 *  Record event _self_ asynchronously in _stream_.
 *  Setting _stream_ to anything other than an instance of CUStream will record on the default stream 0.
 */
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
    if (status == CUDA_ERROR_INVALID_VALUE) {
        RAISE_CU_STD_ERROR(status, "Failed to record event: cuEventRecord() has been called and has not been recorded yet.");
    } else if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to record event.");
    }
    return self;
}

/*  call-seq: event.synchronize    ->    self
 *
 *  Block until _self_ has been recorded.
 */
static VALUE event_synchronize(VALUE self)
{
    CUevent* p;
    Data_Get_Struct(self, CUevent, p);
    CUresult status = cuEventSynchronize(*p);
    // TODO: Handle status == CUDA_ERROR_INVALID_VALUE
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to synchronize event.");
    }
    return self;
}

/*  call-seq: event.elapsed_time(event_start, event_end)    ->    Numeric
 *
 *  Return the elapsed time (ms) from _event_start_ to _event_end_.
 */
static VALUE event_elapsed_time(VALUE klass, VALUE event_start, VALUE event_end)
{
    CUevent* pevent_start;
    CUevent* pevent_end;
    Data_Get_Struct(event_start, CUevent, pevent_start);
    Data_Get_Struct(event_end, CUevent, pevent_end);
    float etime;
    CUresult status = cuEventElapsedTime(&etime, *pevent_start, *pevent_end);
    if (status == CUDA_ERROR_NOT_READY) {
        RAISE_CU_STD_ERROR(status, "Failed to get elapsed time of events: either event has not been recorded yet.");
    } else if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get elapsed time of events.");
    }
    return DBL2NUM(etime);
}

// }}}


// {{{ CUtexref

static VALUE texref_alloc(VALUE klass)
{
    CUtexref* p = new CUtexref;
    return Data_Wrap_Struct(klass, 0, generic_free<CUtexref>, p);
}

static VALUE texref_initialize(VALUE self)
{
    return self;
}

/*  call-seq: texref.create    ->    self
 *
 *  Create a texture reference and set _self_ to this texture reference.
 *
 *  Note: This method is *deprecated*.
 */
static VALUE texref_create(VALUE self)
{
    rb_warn("CUTexRef#create is deprecated.");
    CUtexref* p;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefCreate(p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to create texture.");
    }
    return self;
}

/*  call-seq: texref.destroy    ->    nil
 *
 *  Destroy the texture reference _self_.
 *
 *  Note: This method is *deprecated*.
 */
static VALUE texref_destroy(VALUE self)
{
    rb_warn("CUTexRef#destroy is deprecated.");
    CUtexref* p;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefDestroy(*p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to destroy texture.");
    }
    return Qnil;
}

/*  call-seq: texref.get_address    ->    CUDevicePtr
 *
 *  Return a CUDevicePtr instance bound to the texture reference.
 */
static VALUE texref_get_address(VALUE self)
{
    CUtexref* ptexref;
    CUdeviceptr* pdevptr;
    Data_Get_Struct(self, CUtexref, ptexref);
    VALUE rb_devptr = rb_class_new_instance(0, NULL, rb_cCUDevicePtr);
    Data_Get_Struct(rb_devptr, CUdeviceptr, pdevptr);
    CUresult status = cuTexRefGetAddress(pdevptr, *ptexref);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get texture address.");
    }
    return rb_devptr;
}

/*  call-seq: texref.get_address_mode(dim)    ->    Fixnum
 *
 *  Return the address mode of the dimension _dim_ (0..2) of _self_.
 */
static VALUE texref_get_address_mode(VALUE self, VALUE dim)
{
    CUtexref* p;
    CUaddress_mode mode;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefGetAddressMode(&mode, *p, FIX2INT(dim));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to get texture address mode: dim = %d.", FIX2INT(dim));
    }
    return INT2FIX(mode);
}

/*  call-seq: texref.get_filter_mode    ->    Fixnum
 *
 *  Return the filter mode of _self_.
 */
static VALUE texref_get_filter_mode(VALUE self)
{
    CUtexref* p;
    CUfilter_mode mode;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefGetFilterMode(&mode, *p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get texture filter mode.");
    }
    return INT2FIX(mode);
}

/*  call-seq: texref.get_flags    ->    Numeric
 *
 *  Return the flags of _self_.
 */
static VALUE texref_get_flags(VALUE self)
{
    CUtexref* p;
    unsigned int flags;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefGetFlags(&flags, *p);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get texture flags.");
    }
    return UINT2NUM(flags);
}

/*  call-seq: texref.set_address(devptr, nbytes)    ->    Numeric
 *
 *  Bind _devptr_ with _nbytes_ to _self_.
 */
static VALUE texref_set_address(VALUE self, VALUE rb_device_ptr, VALUE nbytes)
{
    CUtexref* ptexref;
    CUdeviceptr* pdevptr;
    size_t offset;
    Data_Get_Struct(self, CUtexref, ptexref);
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevptr);
    CUresult status = cuTexRefSetAddress(&offset, *ptexref, *pdevptr, NUM2UINT(nbytes));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set texture address: nbytes = %u.", NUM2UINT(nbytes));
    }
    return SIZET2NUM(offset);
}

/*  call-seq: texref.set_address_mode(dim, mode)    ->    self
 *
 *  Set the address mode of _self_ with _dim_ and _mode_ (CUAddressMode).
 */
static VALUE texref_set_address_mode(VALUE self, VALUE dim, VALUE mode)
{
    CUtexref* p;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefSetAddressMode(*p, FIX2INT(dim), static_cast<CUaddress_mode>(FIX2INT(mode)));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set texture address mode: dim = %d, mode = %d", FIX2INT(dim), FIX2INT(mode));
    }
    return self;
}

/*  call-seq: texref.set_filter_mode(mode)    ->    self
 *
 *  Set the filter mode of _self_ with _mode_ (CUFilterMode).
 */
static VALUE texref_set_filter_mode(VALUE self, VALUE mode)
{
    CUtexref* p;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefSetFilterMode(*p, static_cast<CUfilter_mode>(FIX2INT(mode)));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set texture filter mode: mode = %d.", FIX2INT(mode));
    }
    return self;
}

/*  call-seq: texref.set_flags(flags)    ->    self
 *
 *  Set the _flags_ (CUTexRefFlags) of _self_.
 */
static VALUE texref_set_flags(VALUE self, VALUE flags)
{
    CUtexref* p;
    Data_Get_Struct(self, CUtexref, p);
    CUresult status = cuTexRefSetFlags(*p, NUM2UINT(flags));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR_FORMATTED(status, "Failed to set texture flags: flags = 0x%x.", NUM2UINT(flags));
    }
    return self;
}

// }}}


// {{{ Memory pointer
static VALUE memory_pointer_alloc(VALUE klass)
{
    MemoryPointer* ppointer = new MemoryPointer;
    ppointer->p = NULL;
    return Data_Wrap_Struct(klass, 0, generic_free<MemoryPointer>, ppointer);
}

static VALUE memory_pointer_initialize(VALUE self)
{
    return self;
}
// }}}


// {{{ Buffer

/*  call-seq: Buffer.new(size, options = {})    ->    Buffer
 *
 *  Create a buffer with _size_ elements.
 *
 *  Options:
 *  * _page_locked_ - Allocate page-locked memory if _:page_locked_ is true. Otherwise, allocate pageable memory.
 *
 *      Buffer.new(10)                       # Allocate 10 elements with pageable memory.
 *      Buffer.new(20, page_locked: true)    # Allocate 20 elements with page-locked memory.
 */
static VALUE ibuffer_initialize(int argc, VALUE* argv, VALUE self)
{
    // This function exists for documentation only.
    rb_notimplement();
    return Qnil;
}

/*  call-seq: Buffer.element_size
 *
 *  Return the size of an element of this Buffer in bytes.
 */
static VALUE ibuffer_element_size(VALUE klass)
{
    rb_notimplement();
    return Qnil;
}

/*  call-seq: buffer.size    ->    Numeric
 *
 *  Return the number of elements in this buffer.
 */
static VALUE ibuffer_size(VALUE self)
{
    rb_notimplement();
    return Qnil;
}

/*  call-seq: buffer.page_locked?    ->    true or false
 *
 *  Return true if this buffer is page-locked allocated.
 *  Otherwise, return false.
 */
static VALUE ibuffer_is_page_locked(VALUE self)
{
    rb_notimplement();
    return Qnil;
}

/*  call-seq: buffer.offset(index)    ->    MemoryPointer
 *
 *  Return the memory pointer of the element at _index_ (0...size) in this buffer.
 */
static VALUE ibuffer_offset(VALUE self, VALUE offset)
{
    rb_notimplement();
    return Qnil;
}

/*  call-seq: buffer[index]    ->    Object
 *
 *  Return the element at _index_ (0...size) in this buffer.
 */
static VALUE ibuffer_element_get(VALUE self, VALUE index)
{
    rb_notimplement();
    return Qnil;
}

/*  call-seq: buffer[index] = value    ->    Object
 *
 *  Set the element at _index_ (0...size) in this buffer to _value_.
 *  Return _value_.
 */
static VALUE ibuffer_element_set(VALUE self, VALUE index, VALUE value)
{
    rb_notimplement();
    return Qnil;
}

static void memory_buffer_free(void* p)
{
    MemoryBuffer* pbuffer = static_cast<MemoryBuffer*>(p);
    if (pbuffer->is_page_locked) {
        cuMemFreeHost(reinterpret_cast<void*>(pbuffer->p));
    } else {
        delete[] pbuffer->p;
    }
    delete pbuffer;
}

static VALUE memory_buffer_alloc(VALUE klass)
{
    MemoryBuffer* pbuffer = new MemoryBuffer;
    pbuffer->size = 0;
    pbuffer->is_page_locked = false;
    pbuffer->p = NULL;
    return Data_Wrap_Struct(klass, 0, memory_buffer_free, pbuffer);
}

static VALUE memory_buffer_element_size(VALUE klass)
{
    return INT2FIX(1);
}

static VALUE memory_buffer_initialize(int argc, VALUE* argv, VALUE self)
{
    if (argc < 1 || argc > 2) {
        rb_raise(rb_eArgError, "wrong number of arguments (%d for 1 or 2).", argc);
    }

    bool use_page_locked = false;
    size_t nbytes = NUM2SIZET(argv[0]);
    if (argc == 2 && CLASS_OF(argv[1]) == rb_cHash) {
        if (rb_hash_aref(argv[1], ID2SYM(rb_intern("page_locked"))) == Qtrue) {
            use_page_locked = true;
        }
    }

    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    pbuffer->size = nbytes;
    if (use_page_locked) {
        CUresult status = cuMemAllocHost(reinterpret_cast<void**>(&pbuffer->p), nbytes);
        if (status != CUDA_SUCCESS) {
            RAISE_CU_STD_ERROR(status, "Failed to allocate page-locked host memory.");
        }
        pbuffer->is_page_locked = true;
    } else {
        pbuffer->p = new char[nbytes];
        pbuffer->is_page_locked = false;
    }
    std::memset(static_cast<void*>(pbuffer->p), 0, pbuffer->size);
    return self;
}

static VALUE memory_buffer_size(VALUE self)
{
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    return SIZET2NUM(pbuffer->size);
}

static VALUE memory_buffer_is_page_locked(VALUE self)
{
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    return to_rb(pbuffer->is_page_locked);
}

static VALUE memory_buffer_offset(VALUE self, VALUE offset)
{
    MemoryBuffer* pbuffer;
    MemoryPointer* ppointer_offset;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    VALUE rb_ppointer_offset = rb_class_new_instance(0, NULL, rb_cMemoryPointer);
    Data_Get_Struct(rb_ppointer_offset, MemoryPointer, ppointer_offset);
    ppointer_offset->p = pbuffer->p + NUM2SIZET(offset);
    return rb_ppointer_offset;
}

static VALUE memory_buffer_element_get(VALUE self, VALUE index)
{
    size_t i = NUM2SIZET(index);
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    int element = static_cast<int>(pbuffer->p[i]);
    return to_rb(element);
}

static VALUE memory_buffer_element_set(VALUE self, VALUE index, VALUE value)
{
    size_t i = NUM2SIZET(index);
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    pbuffer->p[i] = static_cast<char>(FIX2INT(value));
    return value;
}

template <typename TElement>
static void buffer_free(void* p)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    TBuffer* pbuffer = static_cast<TBuffer*>(p);
    if (pbuffer->is_page_locked) {
        cuMemFreeHost(reinterpret_cast<void*>(pbuffer->p));
    } else {
        delete[] pbuffer->p;
    }
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
static VALUE buffer_element_size(VALUE klass)
{
    return INT2FIX(sizeof(TElement));
}
typedef VALUE (*BufferElementSizeFunctionType)(VALUE);

template <typename TElement>
static VALUE buffer_initialize(int argc, VALUE* argv, VALUE self)
{
    if (argc <= 0 || argc >= 3) {
        rb_raise(rb_eArgError, "wrong number of arguments (%d for 1 or 2).", argc);
    }

    bool use_page_locked = false;
    VALUE n = NUM2SIZET(argv[0]);
    if (argc == 2 && CLASS_OF(argv[1]) == rb_cHash) {
        if (rb_hash_aref(argv[1], ID2SYM(rb_intern("page_locked"))) == Qtrue) {
            use_page_locked = true;
        }
    }

    typedef struct TypedBuffer<TElement> TBuffer;
    TBuffer* pbuffer;
    Data_Get_Struct(self, TBuffer, pbuffer);
    pbuffer->size = n*sizeof(TElement);
    if (use_page_locked) {
        CUresult status = cuMemAllocHost(reinterpret_cast<void**>(&pbuffer->p), n*sizeof(TElement));
        if (status != CUDA_SUCCESS) {
            RAISE_CU_STD_ERROR(status, "Failed to allocate page-locked host memory.");
        }
        pbuffer->is_page_locked = true;
    } else {
        pbuffer->p = reinterpret_cast<char*>(new TElement[n]);
        pbuffer->is_page_locked = false;
    }
    std::memset(static_cast<void*>(pbuffer->p), 0, pbuffer->size);
    return self;
}
typedef VALUE (*BufferInitializeFunctionType)(int, VALUE*, VALUE);

template <typename TElement>
static VALUE buffer_size(VALUE self)
{
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    return SIZET2NUM(pbuffer->size / sizeof(TElement));
}
typedef VALUE (*BufferSizeFunctionType)(VALUE);

template <typename TElement>
static VALUE buffer_is_page_locked(VALUE self)
{
    MemoryBuffer* pbuffer;
    Data_Get_Struct(self, MemoryBuffer, pbuffer);
    return to_rb(pbuffer->is_page_locked);
}
typedef VALUE (*BufferIsPageLocked)(VALUE);

template <typename TElement>
static VALUE buffer_offset(VALUE self, VALUE offset)
{
    typedef struct TypedBuffer<TElement> TBuffer;
    TBuffer* pbuffer;
    MemoryPointer* ppointer_offset;
    Data_Get_Struct(self, TBuffer, pbuffer);
    VALUE rb_ppointer_offset = rb_class_new_instance(0, NULL, rb_cMemoryPointer);
    Data_Get_Struct(rb_ppointer_offset, MemoryPointer, ppointer_offset);
    ppointer_offset->p = pbuffer->p + NUM2SIZET(offset)*sizeof(TElement);
    return rb_ppointer_offset;
}
typedef VALUE (*BufferOffsetFunctionType)(VALUE, VALUE);

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


// {{{ Memory

/*  call-seq: memcpy_htod(dst_devptr, src_mem, nbytes)    ->    nil
 *
 *  Copy _nbytes_ from host memory at _src_mem_ to device memory at _dst_devptr_.
 */
static VALUE memcpy_htod(VALUE self, VALUE rb_device_ptr, VALUE rb_memory, VALUE nbytes)
{
    CUdeviceptr* pdevice_ptr;
    MemoryPointer* pmem;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryPointer, pmem);
    CUresult status = cuMemcpyHtoD(*pdevice_ptr, static_cast<void*>(pmem->p), NUM2UINT(nbytes));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to copy memory from host to device.");
    }
    return Qnil;
}

/*  call-seq: memcpy_htod_async(dst_devptr, src_mem, nbytes, stream)    ->    nil
 *
 *  Copy _nbytes_ from host memory at _src_mem_ to device memory at _dst_devptr_ in _stream_ asynchronously.
 *
 *  Note: The _src_mem_ should be *page-locked* memory.
 */
static VALUE memcpy_htod_async(VALUE self, VALUE rb_device_ptr, VALUE rb_memory, VALUE nbytes, VALUE rb_stream)
{
    CUdeviceptr* pdevice_ptr;
    MemoryPointer* pmem;
    CUstream* pstream;
    CUstream stream0 = 0;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryPointer, pmem);
    if (CLASS_OF(rb_stream) == rb_cCUStream) {
        Data_Get_Struct(rb_stream, CUstream, pstream);
    } else {
        pstream = &stream0;
    }
    CUresult status = cuMemcpyHtoDAsync(*pdevice_ptr, static_cast<void*>(pmem->p), NUM2UINT(nbytes), *pstream);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to copy memory asynchronously from host to device.");
    }
    return Qnil;
}

/*  call-seq: memcpy_dtoh(dst_mem, src_devptr, nbytes)    ->    nil
 *
 *  Copy _nbytes_ from device memory at _src_devptr_ to host memory at _dst_mem_.
 */
static VALUE memcpy_dtoh(VALUE self, VALUE rb_memory, VALUE rb_device_ptr, VALUE nbytes)
{
    MemoryPointer* pmem;
    CUdeviceptr* pdevice_ptr;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryPointer, pmem);
    CUresult status = cuMemcpyDtoH(static_cast<void*>(pmem->p), *pdevice_ptr, NUM2UINT(nbytes));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to copy memory from device to host.");
    }
    return Qnil;
}

/*  call-seq: memcpy_dtoh_async(dst_mem, src_devptr, nbytes, stream)    ->    nil
 *
 *  Copy _nbytes_ from device memory at _src_devptr_ to host memory at _dst_mem_ in _stream_ asynchronously.
 *
 *  Note: The _dst_mem_ should be *page-locked* memory.
 */
static VALUE memcpy_dtoh_async(VALUE self, VALUE rb_memory, VALUE rb_device_ptr, VALUE nbytes, VALUE rb_stream)
{
    MemoryPointer* pmem;
    CUdeviceptr* pdevice_ptr;
    CUstream* pstream;
    CUstream stream0 = 0;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryPointer, pmem);
    if (CLASS_OF(rb_stream) == rb_cCUStream) {
        Data_Get_Struct(rb_stream, CUstream, pstream);
    } else {
        pstream = &stream0;
    }
    CUresult status = cuMemcpyDtoHAsync(static_cast<void*>(pmem->p), *pdevice_ptr, NUM2UINT(nbytes), *pstream);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to copy memory asynchronously from device to host.");
    }
    return Qnil;
}

/*  call-seq: memcpy_dtod(dst_devptr, src_devptr, nbytes)    ->    nil
 *
 *  Copy _nbytes_ from device memory at _src_devptr_ to device memory at _dst_devptr_ asynchronously.
 */
static VALUE memcpy_dtod(VALUE self, VALUE rb_device_ptr_dst, VALUE rb_device_ptr_src, VALUE nbytes)
{
    CUdeviceptr* dst;
    CUdeviceptr* src;
    Data_Get_Struct(rb_device_ptr_dst, CUdeviceptr, dst);
    Data_Get_Struct(rb_device_ptr_src, CUdeviceptr, src);
    CUresult status = cuMemcpyDtoD(*dst, *src, NUM2UINT(nbytes));
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to copy memory from device to device.");
    }
    return Qnil;
}

/*  call-seq: memcpy_dtod_async(dst_devptr, src_devptr, nbytes, stream)    ->    nil
 *
 *  Copy _nbytes_ from device memory at _src_devptr_ to device memory at _dst_devptr_ in _stream_ asynchronously.
 */
static VALUE memcpy_dtod_async(VALUE self, VALUE rb_device_ptr_dst, VALUE rb_device_ptr_src, VALUE nbytes, VALUE rb_stream)
{
    CUdeviceptr* dst;
    CUdeviceptr* src;
    CUstream *pstream;
    CUstream stream0 = 0;
    Data_Get_Struct(rb_device_ptr_dst, CUdeviceptr, dst);
    Data_Get_Struct(rb_device_ptr_src, CUdeviceptr, src);
    if (CLASS_OF(rb_stream) == rb_cCUStream) {
        Data_Get_Struct(rb_stream, CUstream, pstream);
    } else {
        pstream = &stream0;
    }
    CUresult status = cuMemcpyDtoDAsync(*dst, *src, NUM2UINT(nbytes), *pstream);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to copy memory asynchronously from device to device.");
    }
    return Qnil;
}

/*  call-seq: mem_get_info    ->    Hash { free:, total: }
 *
 *  Return a hash { free:, total: } with the amount of free and total device memory in bytes.
 */
static VALUE mem_get_info(VALUE self)
{
    size_t free_memory;
    size_t total_memory;
    CUresult status = cuMemGetInfo(&free_memory, &total_memory);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to get memory information.");
    }
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("free")), UINT2NUM(free_memory));
    rb_hash_aset(h, ID2SYM(rb_intern("total")), UINT2NUM(total_memory));
    return h;
}

// }}}


// {{{ Driver

/*  call-seq: driver_get_version    ->    Fixnum
 *
 *  Return the version number of the installed CUDA driver.
 */
static VALUE driver_get_version()
{
    int v;
    cuDriverGetVersion(&v);
    return INT2FIX(v);
}

// }}}


// {{{ Doc

/*  Document-class: SGC::CU::MemoryBuffer
 *  See IBuffer and IBuffer::ClassMethods.
 *
 *  Note: ELEMENT_SIZE is *deprecated*. Use MemoryBuffer.element_size.
 */

/*  Document-class: SGC::CU::Int32Buffer
 *  See IBuffer and IBuffer::ClassMethods.
 *
 *  Note: ELEMENT_SIZE is *deprecated*. Use Int32Buffer.element_size.
 */

/*  Document-class: SGC::CU::Int64Buffer
 *  See IBuffer and IBuffer::ClassMethods.
 *
 *  Note: ELEMENT_SIZE is *deprecated*. Use Int64Buffer.element_size.
 */

/*  Document-class: SGC::CU::Float32Buffer
 *  See IBuffer and IBuffer::ClassMethods.
 *
 *  Note: ELEMENT_SIZE is *deprecated*. Use Float32Buffer.element_size.
 */

/*  Document-class: SGC::CU::Float64Buffer
 *  See IBuffer and IBuffer::ClassMethods.
 *
 *  Note: ELEMENT_SIZE is *deprecated*. Use Float64Buffer.element_size.
 */

 // }}}


extern "C" void Init_rubycu()
{
    rb_mSGC = rb_define_module("SGC");
    rb_mCU  = rb_define_module_under(rb_mSGC, "CU");

    rb_cCUDevice = rb_define_class_under(rb_mCU, "CUDevice", rb_cObject);
    rb_define_singleton_method(rb_cCUDevice, "get_count", RUBY_METHOD_FUNC(device_get_count), 0);
    rb_define_singleton_method(rb_cCUDevice, "get", RUBY_METHOD_FUNC(device_get), 1);
    rb_define_alloc_func(rb_cCUDevice, device_alloc);
    rb_define_method(rb_cCUDevice, "initialize", RUBY_METHOD_FUNC(device_initialize), -1);
    rb_define_method(rb_cCUDevice, "get_name", RUBY_METHOD_FUNC(device_get_name), 0);
    rb_define_method(rb_cCUDevice, "compute_capability", RUBY_METHOD_FUNC(device_compute_capability), 0);
    rb_define_method(rb_cCUDevice, "get_attribute", RUBY_METHOD_FUNC(device_get_attribute), 1);
    rb_define_method(rb_cCUDevice, "get_properties", RUBY_METHOD_FUNC(device_get_properties), 0);
    rb_define_method(rb_cCUDevice, "total_mem", RUBY_METHOD_FUNC(device_total_mem), 0);

    rb_cCUComputeMode = rb_define_class_under(rb_mCU, "CUComputeMode", rb_cObject);
    rb_define_const(rb_cCUComputeMode, "DEFAULT", INT2FIX(CU_COMPUTEMODE_DEFAULT));
    rb_define_const(rb_cCUComputeMode, "EXCLUSIVE", INT2FIX(CU_COMPUTEMODE_EXCLUSIVE));
    rb_define_const(rb_cCUComputeMode, "PROHIBITED", INT2FIX(CU_COMPUTEMODE_PROHIBITED));

    rb_cCUDeviceAttribute = rb_define_class_under(rb_mCU, "CUDeviceAttribute", rb_cObject);
    rb_define_const(rb_cCUDeviceAttribute, "MAX_THREADS_PER_BLOCK", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_BLOCK_DIM_X", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_BLOCK_DIM_Y", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_BLOCK_DIM_Z", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_GRID_DIM_X", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_GRID_DIM_Y", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_GRID_DIM_Z", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_REGISTERS_PER_BLOCK", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_SHARED_MEMORY_PER_BLOCK", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttribute, "TOTAL_CONSTANT_MEMORY", INT2FIX(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY));
    rb_define_const(rb_cCUDeviceAttribute, "WARP_SIZE", INT2FIX(CU_DEVICE_ATTRIBUTE_WARP_SIZE));
    rb_define_const(rb_cCUDeviceAttribute, "MAX_PITCH", INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_PITCH));
    rb_define_const(rb_cCUDeviceAttribute, "CLOCK_RATE", INT2FIX(CU_DEVICE_ATTRIBUTE_CLOCK_RATE));
    rb_define_const(rb_cCUDeviceAttribute, "TEXTURE_ALIGNMENT", INT2FIX(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT));
    rb_define_const(rb_cCUDeviceAttribute, "GPU_OVERLAP", INT2FIX(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP));
    rb_define_const(rb_cCUDeviceAttribute, "MULTIPROCESSOR_COUNT", INT2FIX(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
    rb_define_const(rb_cCUDeviceAttribute, "KERNEL_EXEC_TIMEOUT", INT2FIX(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT));
    rb_define_const(rb_cCUDeviceAttribute, "INTEGRATED", INT2FIX(CU_DEVICE_ATTRIBUTE_INTEGRATED));
    rb_define_const(rb_cCUDeviceAttribute, "CAN_MAP_HOST_MEMORY", INT2FIX(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY));
    rb_define_const(rb_cCUDeviceAttribute, "COMPUTE_MODE", INT2FIX(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE1D_WIDTH", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_WIDTH", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE3D_WIDTH", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_HEIGHT", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE3D_HEIGHT", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE3D_DEPTH", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_ARRAY_WIDTH", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_ARRAY_HEIGHT", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT));
    rb_define_const(rb_cCUDeviceAttribute, "MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES));
    rb_define_const(rb_cCUDeviceAttribute, "SURFACE_ALIGNMENT", INT2FIX(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT));
    rb_define_const(rb_cCUDeviceAttribute, "CONCURRENT_KERNELS", INT2FIX(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS));
    rb_define_const(rb_cCUDeviceAttribute, "ECC_ENABLED", INT2FIX(CU_DEVICE_ATTRIBUTE_ECC_ENABLED));
    rb_define_const(rb_cCUDeviceAttribute, "PCI_BUS_ID", INT2FIX(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID));
    rb_define_const(rb_cCUDeviceAttribute, "PCI_DEVICE_ID", INT2FIX(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID));
    rb_define_const(rb_cCUDeviceAttribute, "TCC_DRIVER", INT2FIX(CU_DEVICE_ATTRIBUTE_TCC_DRIVER));

    rb_cCUContext = rb_define_class_under(rb_mCU, "CUContext", rb_cObject);
    rb_define_alloc_func(rb_cCUContext, context_alloc);
    rb_define_method(rb_cCUContext, "initialize", RUBY_METHOD_FUNC(context_initialize), -1);
    rb_define_method(rb_cCUContext, "create", RUBY_METHOD_FUNC(context_create), -1);
    rb_define_method(rb_cCUContext, "destroy", RUBY_METHOD_FUNC(context_destroy), 0);
    rb_define_method(rb_cCUContext, "attach", RUBY_METHOD_FUNC(context_attach), -1);
    rb_define_method(rb_cCUContext, "detach", RUBY_METHOD_FUNC(context_detach), 0);
    rb_define_method(rb_cCUContext, "push_current", RUBY_METHOD_FUNC(context_push_current), 0);
    rb_define_method(rb_cCUContext, "get_api_version", RUBY_METHOD_FUNC(context_get_api_version), 0);
    rb_define_singleton_method(rb_cCUContext, "get_device", RUBY_METHOD_FUNC(context_get_device), 0);
    rb_define_singleton_method(rb_cCUContext, "get_limit", RUBY_METHOD_FUNC(context_get_limit), 1);
    rb_define_singleton_method(rb_cCUContext, "set_limit", RUBY_METHOD_FUNC(context_set_limit), 2);
    rb_define_singleton_method(rb_cCUContext, "get_cache_config", RUBY_METHOD_FUNC(context_get_cache_config), 0);
    rb_define_singleton_method(rb_cCUContext, "set_cache_config", RUBY_METHOD_FUNC(context_set_cache_config), 1);
    rb_define_singleton_method(rb_cCUContext, "get_api_version", RUBY_METHOD_FUNC(context_get_api_version_singleton), 0);
    rb_define_singleton_method(rb_cCUContext, "pop_current", RUBY_METHOD_FUNC(context_pop_current), 0);
    rb_define_singleton_method(rb_cCUContext, "synchronize", RUBY_METHOD_FUNC(context_synchronize), 0);

    rb_cCUContextFlags = rb_define_class_under(rb_mCU, "CUContextFlags", rb_cObject);
    rb_define_const(rb_cCUContextFlags, "SCHED_AUTO", INT2FIX(CU_CTX_SCHED_AUTO));
    rb_define_const(rb_cCUContextFlags, "SCHED_SPIN", INT2FIX(CU_CTX_SCHED_SPIN));
    rb_define_const(rb_cCUContextFlags, "SCHED_YIELD", INT2FIX(CU_CTX_SCHED_YIELD));
    rb_define_const(rb_cCUContextFlags, "BLOCKING_SYNC", INT2FIX(CU_CTX_BLOCKING_SYNC));
    rb_define_const(rb_cCUContextFlags, "MAP_HOST", INT2FIX(CU_CTX_MAP_HOST));
    rb_define_const(rb_cCUContextFlags, "LMEM_RESIZE_TO_MAX", INT2FIX(CU_CTX_LMEM_RESIZE_TO_MAX));

    rb_cCULimit = rb_define_class_under(rb_mCU, "CULimit", rb_cObject);
    rb_define_const(rb_cCULimit, "STACK_SIZE", INT2FIX(CU_LIMIT_STACK_SIZE));
    rb_define_const(rb_cCULimit, "PRINTF_FIFO_SIZE", INT2FIX(CU_LIMIT_PRINTF_FIFO_SIZE));
    rb_define_const(rb_cCULimit, "MALLOC_HEAP_SIZE", INT2FIX(CU_LIMIT_MALLOC_HEAP_SIZE));

    rb_cCUModule = rb_define_class_under(rb_mCU, "CUModule", rb_cObject);
    rb_define_alloc_func(rb_cCUModule, module_alloc);
    rb_define_method(rb_cCUModule, "initialize", RUBY_METHOD_FUNC(module_initialize), -1);
    rb_define_method(rb_cCUModule, "load", RUBY_METHOD_FUNC(module_load), 1);
    rb_define_method(rb_cCUModule, "load_data", RUBY_METHOD_FUNC(module_load_data), 1);
    rb_define_method(rb_cCUModule, "unload", RUBY_METHOD_FUNC(module_unload), 0);
    rb_define_method(rb_cCUModule, "get_function", RUBY_METHOD_FUNC(module_get_function), 1);
    rb_define_method(rb_cCUModule, "get_global", RUBY_METHOD_FUNC(module_get_global), 1);
    rb_define_method(rb_cCUModule, "get_texref", RUBY_METHOD_FUNC(module_get_texref), 1);

    rb_cCUDevicePtr = rb_define_class_under(rb_mCU, "CUDevicePtr", rb_cObject);
    rb_define_alloc_func(rb_cCUDevicePtr, device_ptr_alloc);
    rb_define_method(rb_cCUDevicePtr, "initialize", RUBY_METHOD_FUNC(device_ptr_initialize), -1);
    rb_define_method(rb_cCUDevicePtr, "offset", RUBY_METHOD_FUNC(device_ptr_offset), 1);
    rb_define_method(rb_cCUDevicePtr, "mem_alloc", RUBY_METHOD_FUNC(device_ptr_mem_alloc), 1);
    rb_define_method(rb_cCUDevicePtr, "mem_free", RUBY_METHOD_FUNC(device_ptr_mem_free), 0);

    rb_cCUFunction = rb_define_class_under(rb_mCU, "CUFunction", rb_cObject);
    rb_define_alloc_func(rb_cCUFunction, function_alloc);
    rb_define_method(rb_cCUFunction, "initialize", RUBY_METHOD_FUNC(function_initialize), -1);
    rb_define_method(rb_cCUFunction, "set_param", RUBY_METHOD_FUNC(function_set_param), -1);
    rb_define_method(rb_cCUFunction, "set_texref", RUBY_METHOD_FUNC(function_set_texref), 1);
    rb_define_method(rb_cCUFunction, "set_block_shape", RUBY_METHOD_FUNC(function_set_block_shape), -1);
    rb_define_method(rb_cCUFunction, "set_shared_size", RUBY_METHOD_FUNC(function_set_shared_size), 1);
    rb_define_method(rb_cCUFunction, "launch", RUBY_METHOD_FUNC(function_launch), 0);
    rb_define_method(rb_cCUFunction, "launch_grid", RUBY_METHOD_FUNC(function_launch_grid), -1);
    rb_define_method(rb_cCUFunction, "launch_grid_async", RUBY_METHOD_FUNC(function_launch_grid_async), -1);
    rb_define_method(rb_cCUFunction, "get_attribute", RUBY_METHOD_FUNC(function_get_attribute), 1);
    rb_define_method(rb_cCUFunction, "set_cache_config", RUBY_METHOD_FUNC(function_set_cache_config), 1);

    rb_cCUFunctionAttribute = rb_define_class_under(rb_mCU, "CUFunctionAttribute", rb_cObject);
    rb_define_const(rb_cCUFunctionAttribute, "MAX_THREADS_PER_BLOCK", INT2FIX(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
    rb_define_const(rb_cCUFunctionAttribute, "SHARED_SIZE_BYTES", INT2FIX(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
    rb_define_const(rb_cCUFunctionAttribute, "CONST_SIZE_BYTES", INT2FIX(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES));
    rb_define_const(rb_cCUFunctionAttribute, "LOCAL_SIZE_BYTES", INT2FIX(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
    rb_define_const(rb_cCUFunctionAttribute, "NUM_REGS", INT2FIX(CU_FUNC_ATTRIBUTE_NUM_REGS));
    rb_define_const(rb_cCUFunctionAttribute, "PTX_VERSION", INT2FIX(CU_FUNC_ATTRIBUTE_PTX_VERSION));
    rb_define_const(rb_cCUFunctionAttribute, "BINARY_VERSION", INT2FIX(CU_FUNC_ATTRIBUTE_BINARY_VERSION));

    rb_cCUFunctionCache = rb_define_class_under(rb_mCU, "CUFunctionCache", rb_cObject);
    rb_define_const(rb_cCUFunctionCache, "PREFER_NONE", INT2FIX(CU_FUNC_CACHE_PREFER_NONE));
    rb_define_const(rb_cCUFunctionCache, "PREFER_SHARED", INT2FIX(CU_FUNC_CACHE_PREFER_SHARED));
    rb_define_const(rb_cCUFunctionCache, "PREFER_L1", INT2FIX(CU_FUNC_CACHE_PREFER_L1));

    rb_cCUStream = rb_define_class_under(rb_mCU, "CUStream", rb_cObject);
    rb_define_alloc_func(rb_cCUStream, stream_alloc);
    rb_define_method(rb_cCUStream, "initialize", RUBY_METHOD_FUNC(stream_initialize), 0);
    rb_define_method(rb_cCUStream, "create", RUBY_METHOD_FUNC(stream_create), 1);
    rb_define_method(rb_cCUStream, "destroy", RUBY_METHOD_FUNC(stream_destroy), 0);
    rb_define_method(rb_cCUStream, "query", RUBY_METHOD_FUNC(stream_query), 0);
    rb_define_method(rb_cCUStream, "synchronize", RUBY_METHOD_FUNC(stream_synchronize), 0);

    rb_cCUEvent = rb_define_class_under(rb_mCU, "CUEvent", rb_cObject);
    rb_define_alloc_func(rb_cCUEvent, event_alloc);
    rb_define_method(rb_cCUEvent, "initialize", RUBY_METHOD_FUNC(event_initialize), 0);
    rb_define_method(rb_cCUEvent, "create", RUBY_METHOD_FUNC(event_create), 1);
    rb_define_method(rb_cCUEvent, "destroy", RUBY_METHOD_FUNC(event_destroy), 0);
    rb_define_method(rb_cCUEvent, "query", RUBY_METHOD_FUNC(event_query), 0);
    rb_define_method(rb_cCUEvent, "record", RUBY_METHOD_FUNC(event_record), 1);
    rb_define_method(rb_cCUEvent, "synchronize", RUBY_METHOD_FUNC(event_synchronize), 0);
    rb_define_singleton_method(rb_cCUEvent, "elapsed_time", RUBY_METHOD_FUNC(event_elapsed_time), 2);

    rb_cCUEventFlags = rb_define_class_under(rb_mCU, "CUEventFlags", rb_cObject);
    rb_define_const(rb_cCUEventFlags, "DEFAULT", INT2FIX(CU_EVENT_DEFAULT));
    rb_define_const(rb_cCUEventFlags, "BLOCKING_SYNC", INT2FIX(CU_EVENT_BLOCKING_SYNC));
    rb_define_const(rb_cCUEventFlags, "DISABLE_TIMING", INT2FIX(CU_EVENT_DISABLE_TIMING));

    rb_cCUAddressMode = rb_define_class_under(rb_mCU, "CUAddressMode", rb_cObject);
    rb_define_const(rb_cCUAddressMode, "WRAP", INT2FIX(CU_TR_ADDRESS_MODE_WRAP));
    rb_define_const(rb_cCUAddressMode, "CLAMP", INT2FIX(CU_TR_ADDRESS_MODE_CLAMP));
    rb_define_const(rb_cCUAddressMode, "MIRROR", INT2FIX(CU_TR_ADDRESS_MODE_MIRROR));
    rb_define_const(rb_cCUAddressMode, "BORDER", INT2FIX(CU_TR_ADDRESS_MODE_BORDER));

    rb_cCUFilterMode = rb_define_class_under(rb_mCU, "CUFilterMode", rb_cObject);
    rb_define_const(rb_cCUFilterMode, "POINT", INT2FIX(CU_TR_FILTER_MODE_POINT));
    rb_define_const(rb_cCUFilterMode, "LINEAR", INT2FIX(CU_TR_FILTER_MODE_LINEAR));

    rb_cCUTexRefFlags = rb_define_class_under(rb_mCU, "CUTexRefFlags", rb_cObject);
    rb_define_const(rb_cCUTexRefFlags, "READ_AS_INTEGER", INT2FIX(CU_TRSF_READ_AS_INTEGER));
    rb_define_const(rb_cCUTexRefFlags, "NORMALIZED_COORDINATES", INT2FIX(CU_TRSF_NORMALIZED_COORDINATES));

    rb_cCUTexRef = rb_define_class_under(rb_mCU, "CUTexRef", rb_cObject);
    rb_define_alloc_func(rb_cCUTexRef, texref_alloc);
    rb_define_method(rb_cCUTexRef, "initialize", RUBY_METHOD_FUNC(texref_initialize), 0);
    rb_define_method(rb_cCUTexRef, "create", RUBY_METHOD_FUNC(texref_create), 0);
    rb_define_method(rb_cCUTexRef, "destroy", RUBY_METHOD_FUNC(texref_destroy), 0);
    rb_define_method(rb_cCUTexRef, "get_address", RUBY_METHOD_FUNC(texref_get_address), 0);
    rb_define_method(rb_cCUTexRef, "get_address_mode", RUBY_METHOD_FUNC(texref_get_address_mode), 1);
    rb_define_method(rb_cCUTexRef, "get_filter_mode", RUBY_METHOD_FUNC(texref_get_filter_mode), 0);
    rb_define_method(rb_cCUTexRef, "get_flags", RUBY_METHOD_FUNC(texref_get_flags), 0);
    rb_define_method(rb_cCUTexRef, "set_address", RUBY_METHOD_FUNC(texref_set_address), 2);
    rb_define_method(rb_cCUTexRef, "set_address_mode", RUBY_METHOD_FUNC(texref_set_address_mode), 2);
    rb_define_method(rb_cCUTexRef, "set_filter_mode", RUBY_METHOD_FUNC(texref_set_filter_mode), 1);
    rb_define_method(rb_cCUTexRef, "set_flags", RUBY_METHOD_FUNC(texref_set_flags), 1);

    rb_cCUResult = rb_define_class_under(rb_mCU, "CUResult", rb_cObject);
    rb_define_const(rb_cCUResult, "SUCCESS", INT2FIX(CUDA_SUCCESS));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_VALUE", INT2FIX(CUDA_ERROR_INVALID_VALUE));
    rb_define_const(rb_cCUResult, "ERROR_OUT_OF_MEMORY", INT2FIX(CUDA_ERROR_OUT_OF_MEMORY));
    rb_define_const(rb_cCUResult, "ERROR_NOT_INITIALIZED", INT2FIX(CUDA_ERROR_NOT_INITIALIZED));
    rb_define_const(rb_cCUResult, "ERROR_DEINITIALIZED", INT2FIX(CUDA_ERROR_DEINITIALIZED));
    rb_define_const(rb_cCUResult, "ERROR_NO_DEVICE", INT2FIX(CUDA_ERROR_NO_DEVICE));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_DEVICE", INT2FIX(CUDA_ERROR_INVALID_DEVICE));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_IMAGE", INT2FIX(CUDA_ERROR_INVALID_IMAGE));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_CONTEXT", INT2FIX(CUDA_ERROR_INVALID_CONTEXT));
    rb_define_const(rb_cCUResult, "ERROR_CONTEXT_ALREADY_CURRENT", INT2FIX(CUDA_ERROR_CONTEXT_ALREADY_CURRENT));
    rb_define_const(rb_cCUResult, "ERROR_MAP_FAILED", INT2FIX(CUDA_ERROR_MAP_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_UNMAP_FAILED", INT2FIX(CUDA_ERROR_UNMAP_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_ARRAY_IS_MAPPED", INT2FIX(CUDA_ERROR_ARRAY_IS_MAPPED));
    rb_define_const(rb_cCUResult, "ERROR_ALREADY_MAPPED", INT2FIX(CUDA_ERROR_ALREADY_MAPPED));
    rb_define_const(rb_cCUResult, "ERROR_NO_BINARY_FOR_GPU", INT2FIX(CUDA_ERROR_NO_BINARY_FOR_GPU));
    rb_define_const(rb_cCUResult, "ERROR_ALREADY_ACQUIRED", INT2FIX(CUDA_ERROR_ALREADY_ACQUIRED));
    rb_define_const(rb_cCUResult, "ERROR_NOT_MAPPED", INT2FIX(CUDA_ERROR_NOT_MAPPED));
    rb_define_const(rb_cCUResult, "ERROR_NOT_MAPPED_AS_ARRAY", INT2FIX(CUDA_ERROR_NOT_MAPPED_AS_ARRAY));
    rb_define_const(rb_cCUResult, "ERROR_NOT_MAPPED_AS_POINTER", INT2FIX(CUDA_ERROR_NOT_MAPPED_AS_POINTER));
    rb_define_const(rb_cCUResult, "ERROR_ECC_UNCORRECTABLE", INT2FIX(CUDA_ERROR_ECC_UNCORRECTABLE));
    rb_define_const(rb_cCUResult, "ERROR_UNSUPPORTED_LIMIT", INT2FIX(CUDA_ERROR_UNSUPPORTED_LIMIT));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_SOURCE", INT2FIX(CUDA_ERROR_INVALID_SOURCE));
    rb_define_const(rb_cCUResult, "ERROR_FILE_NOT_FOUND", INT2FIX(CUDA_ERROR_FILE_NOT_FOUND));
    rb_define_const(rb_cCUResult, "ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", INT2FIX(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND));
    rb_define_const(rb_cCUResult, "ERROR_SHARED_OBJECT_INIT_FAILED", INT2FIX(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_OPERATING_SYSTEM", INT2FIX(CUDA_ERROR_OPERATING_SYSTEM));
    rb_define_const(rb_cCUResult, "ERROR_INVALID_HANDLE", INT2FIX(CUDA_ERROR_INVALID_HANDLE));
    rb_define_const(rb_cCUResult, "ERROR_NOT_FOUND", INT2FIX(CUDA_ERROR_NOT_FOUND));
    rb_define_const(rb_cCUResult, "ERROR_NOT_READY", INT2FIX(CUDA_ERROR_NOT_READY));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_FAILED", INT2FIX(CUDA_ERROR_LAUNCH_FAILED));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_OUT_OF_RESOURCES", INT2FIX(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_TIMEOUT", INT2FIX(CUDA_ERROR_LAUNCH_TIMEOUT));
    rb_define_const(rb_cCUResult, "ERROR_LAUNCH_INCOMPATIBLE_TEXTURING" , INT2FIX(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING));
    rb_define_const(rb_cCUResult, "ERROR_UNKNOWN", INT2FIX(CUDA_ERROR_UNKNOWN));

    rb_eCUStandardError = rb_define_class_under(rb_mCU, "CUStandardError", rb_eStandardError);

    rb_eCUDeviceError               = rb_define_class_under(rb_mCU, "CUDeviceError", rb_eCUStandardError);
    rb_eCUDeviceNotInitializedError = rb_define_class_under(rb_mCU, "CUDeviceNotInitializedError", rb_eCUDeviceError);
    rb_eCUDeviceDeinitializedError  = rb_define_class_under(rb_mCU, "CUDeviceDeinitializedError", rb_eCUDeviceError);
    rb_eCUNoDeviceError             = rb_define_class_under(rb_mCU, "CUNoDeviceError", rb_eCUDeviceError);
    rb_eCUInvalidDeviceError        = rb_define_class_under(rb_mCU, "CUInvalidDeviceError", rb_eCUDeviceError);

    rb_eCUMapError                = rb_define_class_under(rb_mCU, "CUMapError", rb_eCUStandardError);
    rb_eCUMapFailedError          = rb_define_class_under(rb_mCU, "CUMapFailedError", rb_eCUMapError);
    rb_eCUUnMapFailedError        = rb_define_class_under(rb_mCU, "CUUnMapFailedError", rb_eCUMapError);
    rb_eCUArrayIsMappedError      = rb_define_class_under(rb_mCU, "CUArrayIsMappedError", rb_eCUMapError);
    rb_eCUAlreadyMappedError      = rb_define_class_under(rb_mCU, "CUAlreadyMappedError", rb_eCUMapError);
    rb_eCUNotMappedError          = rb_define_class_under(rb_mCU, "CUNotMappedError", rb_eCUMapError);
    rb_eCUNotMappedAsArrayError   = rb_define_class_under(rb_mCU, "CUNotMappedAsArrayError", rb_eCUMapError);
    rb_eCUNotMappedAsPointerError = rb_define_class_under(rb_mCU, "CUNotMappedAsPointerError", rb_eCUMapError);

    rb_eCUContextError               = rb_define_class_under(rb_mCU, "CUContextError", rb_eCUStandardError);
    rb_eCUInvalidContextError        = rb_define_class_under(rb_mCU, "CUInvalidContextError", rb_eCUContextError);
    rb_eCUContextAlreadyCurrentError = rb_define_class_under(rb_mCU, "CUContextAlreadyCurrentError", rb_eCUContextError);
    rb_eCUUnsupportedLimitError      = rb_define_class_under(rb_mCU, "CUUnsupportedLimitError", rb_eCUContextError);

    rb_eCULaunchError                      = rb_define_class_under(rb_mCU, "CULaunchError", rb_eCUStandardError);
    rb_eCULaunchFailedError                = rb_define_class_under(rb_mCU, "CULaunchFailedError", rb_eCULaunchError);
    rb_eCULaunchOutOfResourcesError        = rb_define_class_under(rb_mCU, "CULaunchOutOfResourcesError", rb_eCULaunchError);
    rb_eCULaunchTimeoutError               = rb_define_class_under(rb_mCU, "CULaunchTimeoutError", rb_eCULaunchError);
    rb_eCULaunchIncompatibleTexturingError = rb_define_class_under(rb_mCU, "CULaunchIncompatibleTexturingError", rb_eCULaunchError);

    rb_eCUParameterError     = rb_define_class_under(rb_mCU, "CUParameterError", rb_eCUStandardError);
    rb_eCUInvalidValueError  = rb_define_class_under(rb_mCU, "CUInvalidValueError", rb_eCUParameterError);
    rb_eCUInvalidHandleError = rb_define_class_under(rb_mCU, "CUInvalidHandleError", rb_eCUParameterError);

    rb_eCUMemoryError      = rb_define_class_under(rb_mCU, "CUMemoryError", rb_eCUStandardError);
    rb_eCUOutOfMemoryError = rb_define_class_under(rb_mCU, "CUOutOfMemoryError", rb_eCUMemoryError);

    rb_eCULibraryError                    = rb_define_class_under(rb_mCU, "CULibraryError", rb_eCUStandardError);
    rb_eCUSharedObjectSymbolNotFoundError = rb_define_class_under(rb_mCU, "CUSharedObjectSymbolNotFoundError", rb_eCULibraryError);
    rb_eCUSharedObjectInitFailedError     = rb_define_class_under(rb_mCU, "CUSharedObjectInitFailedError", rb_eCULibraryError);

    rb_eCUHardwareError         = rb_define_class_under(rb_mCU, "CUHardwareError", rb_eCUStandardError);
    rb_eCUECCUncorrectableError = rb_define_class_under(rb_mCU, "CUECCUncorrectableError", rb_eCUHardwareError);

    rb_eCUFileError           = rb_define_class_under(rb_mCU, "CUFileError", rb_eCUStandardError);
    rb_eCUNoBinaryForGPUError = rb_define_class_under(rb_mCU, "CUNoBinaryForGPUError", rb_eCUFileError);
    rb_eCUFileNotFoundError   = rb_define_class_under(rb_mCU, "CUFileNotFoundError", rb_eCUFileError);
    rb_eCUInvalidSourceError  = rb_define_class_under(rb_mCU, "CUInvalidSourceError", rb_eCUFileError);
    rb_eCUInvalidImageError   = rb_define_class_under(rb_mCU, "CUInvalidImageError", rb_eCUFileError);

    rb_eCUReferenceError         = rb_define_class_under(rb_mCU, "CUReferenceError", rb_eCUStandardError);
    rb_eCUReferenceNotFoundError = rb_define_class_under(rb_mCU, "CUReferenceNotFoundError", rb_eCUReferenceError);

    rb_eCUOtherError           = rb_define_class_under(rb_mCU, "CUOtherError", rb_eCUStandardError);
    rb_eCUAlreadyAcquiredError = rb_define_class_under(rb_mCU, "CUAlreadyAcquiredError", rb_eCUOtherError);
    rb_eCUNotReadyError        = rb_define_class_under(rb_mCU, "CUNotReadyError", rb_eCUOtherError);
    rb_eCUOperatingSystemError = rb_define_class_under(rb_mCU, "CUOperatingSystemError", rb_eCUOtherError);

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
    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_OPERATING_SYSTEM), rb_eCUOperatingSystemError);

    rb_hash_aset(rb_error_class_by_enum, INT2FIX(CUDA_ERROR_UNKNOWN), rb_eCUUnknownError);

    rb_cMemoryPointer = rb_define_class_under(rb_mCU, "MemoryPointer", rb_cObject);
    rb_define_alloc_func(rb_cMemoryPointer, memory_pointer_alloc);
    rb_define_method(rb_cMemoryPointer, "initialize", RUBY_METHOD_FUNC(memory_pointer_initialize), 0);

    rb_mIBuffer = rb_define_module_under(rb_mCU, "IBuffer");
    rb_define_singleton_method(rb_mIBuffer, "included", RUBY_METHOD_FUNC(module_included_classmethods_hook), 1);
    rb_define_method(rb_mIBuffer, "initialize", RUBY_METHOD_FUNC(ibuffer_initialize), -1);
    rb_define_method(rb_mIBuffer, "size", RUBY_METHOD_FUNC(ibuffer_size), 0);
    rb_define_method(rb_mIBuffer, "page_locked?", RUBY_METHOD_FUNC(ibuffer_is_page_locked), 0);
    rb_define_method(rb_mIBuffer, "offset", RUBY_METHOD_FUNC(ibuffer_offset), 1);
    rb_define_method(rb_mIBuffer, "[]", RUBY_METHOD_FUNC(ibuffer_element_get), 1);
    rb_define_method(rb_mIBuffer, "[]=", RUBY_METHOD_FUNC(ibuffer_element_set), 2);

    rb_mIBufferClassMethods = rb_define_module_under(rb_mIBuffer, "ClassMethods");
    rb_define_method(rb_mIBufferClassMethods, "element_size", RUBY_METHOD_FUNC(ibuffer_element_size), 0);

    rb_cMemoryBuffer = rb_define_class_under(rb_mCU, "MemoryBuffer", rb_cMemoryPointer);
    rb_include_module(rb_cMemoryBuffer, rb_mIBuffer);
    module_included_classmethods_hook(rb_mIBuffer, rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cMemoryBuffer, memory_buffer_alloc);
    rb_define_singleton_method(rb_cMemoryBuffer, "element_size", RUBY_METHOD_FUNC(memory_buffer_element_size), 0);
    rb_define_method(rb_cMemoryBuffer, "initialize", RUBY_METHOD_FUNC(memory_buffer_initialize), -1);
    rb_define_method(rb_cMemoryBuffer, "size", RUBY_METHOD_FUNC(memory_buffer_size), 0);
    rb_define_method(rb_cMemoryBuffer, "page_locked?", RUBY_METHOD_FUNC(memory_buffer_is_page_locked), 0);
    rb_define_method(rb_cMemoryBuffer, "offset", RUBY_METHOD_FUNC(memory_buffer_offset), 1);
    rb_define_method(rb_cMemoryBuffer, "[]", RUBY_METHOD_FUNC(memory_buffer_element_get), 1);
    rb_define_method(rb_cMemoryBuffer, "[]=", RUBY_METHOD_FUNC(memory_buffer_element_set), 2);

    rb_cInt32Buffer = rb_define_class_under(rb_mCU, "Int32Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cInt32Buffer, buffer_alloc<int>);
    rb_define_const(rb_cInt32Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(int)));
    rb_define_singleton_method(rb_cInt32Buffer, "element_size", RUBY_METHOD_FUNC(static_cast<BufferElementSizeFunctionType>(&buffer_element_size<int>)), 0);
    rb_define_method(rb_cInt32Buffer, "initialize", RUBY_METHOD_FUNC(static_cast<BufferInitializeFunctionType>(&buffer_initialize<int>)) , -1);
    rb_define_method(rb_cInt32Buffer, "size", RUBY_METHOD_FUNC(static_cast<BufferSizeFunctionType>(&buffer_size<int>)), 0);
    rb_define_method(rb_cInt32Buffer, "page_locked?", RUBY_METHOD_FUNC(static_cast<BufferIsPageLocked>(&buffer_is_page_locked<int>)), 0);
    rb_define_method(rb_cInt32Buffer, "offset", RUBY_METHOD_FUNC(static_cast<BufferOffsetFunctionType>(&buffer_offset<int>)), 1);
    rb_define_method(rb_cInt32Buffer, "[]", RUBY_METHOD_FUNC(static_cast<BufferElementGetFunctionType>(&buffer_element_get<int>)), 1);
    rb_define_method(rb_cInt32Buffer, "[]=", RUBY_METHOD_FUNC(static_cast<BufferElementSetFunctionType>(&buffer_element_set<int>)), 2);

    rb_cInt64Buffer = rb_define_class_under(rb_mCU, "Int64Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cInt64Buffer, buffer_alloc<long>);
    rb_define_const(rb_cInt64Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(long)));
    rb_define_singleton_method(rb_cInt64Buffer, "element_size", RUBY_METHOD_FUNC(static_cast<BufferElementSizeFunctionType>(&buffer_element_size<long>)), 0);
    rb_define_method(rb_cInt64Buffer, "initialize", RUBY_METHOD_FUNC(static_cast<BufferInitializeFunctionType>(&buffer_initialize<long>)) , -1);
    rb_define_method(rb_cInt64Buffer, "size", RUBY_METHOD_FUNC(static_cast<BufferSizeFunctionType>(&buffer_size<long>)), 0);
    rb_define_method(rb_cInt64Buffer, "page_locked?", RUBY_METHOD_FUNC(static_cast<BufferIsPageLocked>(&buffer_is_page_locked<long>)), 0);
    rb_define_method(rb_cInt64Buffer, "offset", RUBY_METHOD_FUNC(static_cast<BufferOffsetFunctionType>(&buffer_offset<long>)), 1);
    rb_define_method(rb_cInt64Buffer, "[]", RUBY_METHOD_FUNC(static_cast<BufferElementGetFunctionType>(&buffer_element_get<long>)), 1);
    rb_define_method(rb_cInt64Buffer, "[]=", RUBY_METHOD_FUNC(static_cast<BufferElementSetFunctionType>(&buffer_element_set<long>)), 2);

    rb_cFloat32Buffer = rb_define_class_under(rb_mCU, "Float32Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cFloat32Buffer, buffer_alloc<float>);
    rb_define_const(rb_cFloat32Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(float)));
    rb_define_singleton_method(rb_cFloat32Buffer, "element_size", RUBY_METHOD_FUNC(static_cast<BufferElementSizeFunctionType>(&buffer_element_size<float>)), 0);
    rb_define_method(rb_cFloat32Buffer, "initialize", RUBY_METHOD_FUNC(static_cast<BufferInitializeFunctionType>(&buffer_initialize<float>)) , -1);
    rb_define_method(rb_cFloat32Buffer, "size", RUBY_METHOD_FUNC(static_cast<BufferSizeFunctionType>(&buffer_size<float>)), 0);
    rb_define_method(rb_cFloat32Buffer, "page_locked?", RUBY_METHOD_FUNC(static_cast<BufferIsPageLocked>(&buffer_is_page_locked<float>)), 0);
    rb_define_method(rb_cFloat32Buffer, "offset", RUBY_METHOD_FUNC(static_cast<BufferOffsetFunctionType>(&buffer_offset<float>)), 1);
    rb_define_method(rb_cFloat32Buffer, "[]", RUBY_METHOD_FUNC(static_cast<BufferElementGetFunctionType>(&buffer_element_get<float>)), 1);
    rb_define_method(rb_cFloat32Buffer, "[]=", RUBY_METHOD_FUNC(static_cast<BufferElementSetFunctionType>(&buffer_element_set<float>)), 2);

    rb_cFloat64Buffer = rb_define_class_under(rb_mCU, "Float64Buffer", rb_cMemoryBuffer);
    rb_define_alloc_func(rb_cFloat64Buffer, buffer_alloc<double>);
    rb_define_const(rb_cFloat64Buffer, "ELEMENT_SIZE", INT2FIX(sizeof(double)));
    rb_define_method(rb_cFloat64Buffer, "initialize", RUBY_METHOD_FUNC(static_cast<BufferInitializeFunctionType>(&buffer_initialize<double>)) , -1);
    rb_define_singleton_method(rb_cFloat64Buffer, "element_size", RUBY_METHOD_FUNC(static_cast<BufferElementSizeFunctionType>(&buffer_element_size<double>)), 0);
    rb_define_method(rb_cFloat64Buffer, "size", RUBY_METHOD_FUNC(static_cast<BufferSizeFunctionType>(&buffer_size<double>)), 0);
    rb_define_method(rb_cFloat64Buffer, "page_locked?", RUBY_METHOD_FUNC(static_cast<BufferIsPageLocked>(&buffer_is_page_locked<double>)), 0);
    rb_define_method(rb_cFloat64Buffer, "offset", RUBY_METHOD_FUNC(static_cast<BufferOffsetFunctionType>(&buffer_offset<double>)), 1);
    rb_define_method(rb_cFloat64Buffer, "[]", RUBY_METHOD_FUNC(static_cast<BufferElementGetFunctionType>(&buffer_element_get<double>)), 1);
    rb_define_method(rb_cFloat64Buffer, "[]=", RUBY_METHOD_FUNC(static_cast<BufferElementSetFunctionType>(&buffer_element_set<double>)), 2);

    rb_define_module_function(rb_mCU, "memcpy_htod", RUBY_METHOD_FUNC(memcpy_htod), 3);
    rb_define_module_function(rb_mCU, "memcpy_dtoh", RUBY_METHOD_FUNC(memcpy_dtoh), 3);
    rb_define_module_function(rb_mCU, "memcpy_dtod", RUBY_METHOD_FUNC(memcpy_dtod), 3);
    rb_define_module_function(rb_mCU, "memcpy_htod_async", RUBY_METHOD_FUNC(memcpy_htod_async), 4);
    rb_define_module_function(rb_mCU, "memcpy_dtoh_async", RUBY_METHOD_FUNC(memcpy_dtoh_async), 4);
    rb_define_module_function(rb_mCU, "memcpy_dtod_async", RUBY_METHOD_FUNC(memcpy_dtod_async), 4);
    rb_define_module_function(rb_mCU, "mem_get_info", RUBY_METHOD_FUNC(mem_get_info), 0);

    rb_define_module_function(rb_mCU, "driver_get_version", RUBY_METHOD_FUNC(driver_get_version), 0);

    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        RAISE_CU_STD_ERROR(status, "Failed to initialize the CUDA driver API.");
    }
}

} // namespace
} // namespace
