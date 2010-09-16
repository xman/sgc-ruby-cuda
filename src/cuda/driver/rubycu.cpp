/*
# Copyright (c) 2010 Chung Shin Yee
#
#       shinyee@speedgocomputing.com
#       http://www.speedgocomputing.com
#       http://github.com/xman/sgc-ruby-cuda
#
# This file is part of sgc-ruby-cuda.
#
# sgc-ruby-cuda is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# sgc-ruby-cuda is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with sgc-ruby-cuda.  If not, see <http://www.gnu.org/licenses/>.
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
static VALUE rb_cCUContextFlagsEnum;
static VALUE rb_cCULimitEnum;
static VALUE rb_cCUModule;
static VALUE rb_cCUFunction;
static VALUE rb_cCUDevicePtr;
static VALUE rb_cCUDeviceAttributeEnum;
static VALUE rb_cCUComputeModeEnum;
static VALUE rb_cCUStream;
// }}}

// {{{ SGC Ruby classes.
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
// }}}


// {{{ CUdevice
static VALUE device_get_count(VALUE klass)
{
    int count;
    cuDeviceGetCount(&count);
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
    cuDeviceGet(p, i);
    return self;
}

static VALUE device_get_name(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    char name[256];
    cuDeviceGetName(name, 256, *p);
    return rb_str_new2(name);
}

static VALUE device_compute_capability(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int major;
    int minor;
    cuDeviceComputeCapability(&major, &minor, *p);
    return rb_ary_new3(2, INT2FIX(major), INT2FIX(minor));
}

static VALUE device_get_attribute(VALUE self, VALUE attribute)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int v;
    cuDeviceGetAttribute(&v, static_cast<CUdevice_attribute>(FIX2INT(attribute)), *p);
    return INT2FIX(v);
}

static VALUE device_total_mem(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    unsigned int nbytes;
    cuDeviceTotalMem(&nbytes, *p);
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
    cuCtxCreate(pcontext, FIX2UINT(flags), *pdevice);
    return self;
}

static VALUE context_destroy(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    cuCtxDestroy(*p);
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
    cuCtxAttach(p, flags);
    return self;
}

static VALUE context_detach(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    cuCtxDetach(*p);
    return Qnil;
}

static VALUE context_push_current(VALUE self)
{
    CUcontext* p;
    Data_Get_Struct(self, CUcontext, p);
    cuCtxPushCurrent(*p);
    return self;
}

static VALUE context_get_device(VALUE klass, VALUE device)
{
    CUdevice* pdevice;
    Data_Get_Struct(device, CUdevice, pdevice);
    cuCtxGetDevice(pdevice);
    return Qnil;
}

static VALUE context_get_limit(VALUE klass, VALUE limit)
{
    CUlimit l = static_cast<CUlimit>(FIX2UINT(limit));
    size_t v = 0;
    cuCtxGetLimit(&v, l);
    return LONG2FIX(v);
}

static VALUE context_set_limit(VALUE klass, VALUE limit, VALUE value)
{
    CUlimit l = static_cast<CUlimit>(FIX2UINT(limit));
    size_t v = NUM2UINT(value);
    cuCtxSetLimit(l, v);
    return Qnil;
}

static VALUE context_pop_current(VALUE klass, VALUE context)
{
    CUcontext* pcontext;
    Data_Get_Struct(context, CUcontext, pcontext);
    cuCtxPopCurrent(pcontext);
    return Qnil;
}

static VALUE context_synchronize(VALUE klass)
{
    cuCtxSynchronize();
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
    cuModuleLoad(p, StringValuePtr(str));
    return self;
}

static VALUE module_unload(VALUE self)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    cuModuleUnload(*p);
    return self;
}

static VALUE module_get_function(VALUE self, VALUE str)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    CUfunction* pfunc = new CUfunction;
    cuModuleGetFunction(pfunc, *p, StringValuePtr(str));
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
    cuModuleGetGlobal(pdevptr, &nbytes, *p, StringValuePtr(str));
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
    size_t n = NUM2ULONG(nbytes);
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
    cuStreamCreate(p, FIX2UINT(flags));
    return self;
}

static VALUE stream_destroy(VALUE self)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    cuStreamDestroy(*p);
    return Qnil;
}

static VALUE stream_query(VALUE self)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    CUresult status = cuStreamQuery(*p);
    if (status == CUDA_SUCCESS)
        return Qtrue;
    return Qfalse;
}

static VALUE stream_synchronize(VALUE self)
{
    CUstream* p;
    Data_Get_Struct(self, CUstream, p);
    cuStreamSynchronize(*p);
    return self;
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
    size_t n = NUM2ULONG(nbytes);
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
    size_t n = NUM2ULONG(nelements);
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
    size_t i = NUM2ULONG(index);
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
    size_t i = NUM2ULONG(index);
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
    size_t nbytes = NUM2ULONG(rb_nbytes);
    cuMemcpyHtoD(*pdevice_ptr, static_cast<void*>(pmem->p), nbytes);
    return Qnil; // TODO: Return the status of the transfer.
}

static VALUE memcpy_dtoh(VALUE self, VALUE rb_memory, VALUE rb_device_ptr, VALUE rb_nbytes)
{
    MemoryBuffer* pmem;
    CUdeviceptr* pdevice_ptr;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_memory, MemoryBuffer, pmem);
    size_t nbytes = NUM2ULONG(rb_nbytes);
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

    rb_cCUComputeModeEnum = rb_define_class_under(rb_mCU, "CUComputeModeEnum", rb_cObject);
    rb_define_const(rb_cCUComputeModeEnum, "DEFAULT"   , INT2FIX(CU_COMPUTEMODE_DEFAULT));
    rb_define_const(rb_cCUComputeModeEnum, "EXCLUSIVE" , INT2FIX(CU_COMPUTEMODE_EXCLUSIVE));
    rb_define_const(rb_cCUComputeModeEnum, "PROHIBITED", INT2FIX(CU_COMPUTEMODE_PROHIBITED));

    rb_cCUDeviceAttributeEnum = rb_define_class_under(rb_mCU, "CUDeviceAttributeEnum", rb_cObject);
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_THREADS_PER_BLOCK"            , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_BLOCK_DIM_X"                  , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_BLOCK_DIM_Y"                  , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_BLOCK_DIM_Z"                  , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_GRID_DIM_X"                   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_GRID_DIM_Y"                   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_GRID_DIM_Z"                   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_REGISTERS_PER_BLOCK"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_SHARED_MEMORY_PER_BLOCK"      , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
    rb_define_const(rb_cCUDeviceAttributeEnum, "TOTAL_CONSTANT_MEMORY"            , INT2FIX(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY));
    rb_define_const(rb_cCUDeviceAttributeEnum, "WARP_SIZE"                        , INT2FIX(CU_DEVICE_ATTRIBUTE_WARP_SIZE));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAX_PITCH"                        , INT2FIX(CU_DEVICE_ATTRIBUTE_MAX_PITCH));
    rb_define_const(rb_cCUDeviceAttributeEnum, "CLOCK_RATE"                       , INT2FIX(CU_DEVICE_ATTRIBUTE_CLOCK_RATE));
    rb_define_const(rb_cCUDeviceAttributeEnum, "TEXTURE_ALIGNMENT"                , INT2FIX(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "GPU_OVERLAP"                      , INT2FIX(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MULTIPROCESSOR_COUNT"             , INT2FIX(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "KERNEL_EXEC_TIMEOUT"              , INT2FIX(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "INTEGRATED"                       , INT2FIX(CU_DEVICE_ATTRIBUTE_INTEGRATED));
    rb_define_const(rb_cCUDeviceAttributeEnum, "CAN_MAP_HOST_MEMORY"              , INT2FIX(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY));
    rb_define_const(rb_cCUDeviceAttributeEnum, "COMPUTE_MODE"                     , INT2FIX(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE1D_WIDTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE2D_WIDTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE3D_WIDTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE2D_HEIGHT"         , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE3D_HEIGHT"         , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE3D_DEPTH"          , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE2D_ARRAY_WIDTH"    , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE2D_ARRAY_HEIGHT"   , INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES", INT2FIX(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES));
    rb_define_const(rb_cCUDeviceAttributeEnum, "SURFACE_ALIGNMENT"                , INT2FIX(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT));
    rb_define_const(rb_cCUDeviceAttributeEnum, "CONCURRENT_KERNELS"               , INT2FIX(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS));
    rb_define_const(rb_cCUDeviceAttributeEnum, "ECC_ENABLED"                      , INT2FIX(CU_DEVICE_ATTRIBUTE_ECC_ENABLED));
    rb_define_const(rb_cCUDeviceAttributeEnum, "PCI_BUS_ID"                       , INT2FIX(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID));
    rb_define_const(rb_cCUDeviceAttributeEnum, "PCI_DEVICE_ID"                    , INT2FIX(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID));

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

    rb_cCUContextFlagsEnum = rb_define_class_under(rb_mCU, "CUContextFlagsEnum", rb_cObject);
    rb_define_const(rb_cCUContextFlagsEnum, "SCHED_AUTO"        , INT2FIX(CU_CTX_SCHED_AUTO));
    rb_define_const(rb_cCUContextFlagsEnum, "SCHED_SPIN"        , INT2FIX(CU_CTX_SCHED_SPIN));
    rb_define_const(rb_cCUContextFlagsEnum, "SCHED_YIELD"       , INT2FIX(CU_CTX_SCHED_YIELD));
    rb_define_const(rb_cCUContextFlagsEnum, "BLOCKING_SYNC"     , INT2FIX(CU_CTX_BLOCKING_SYNC));
    rb_define_const(rb_cCUContextFlagsEnum, "MAP_HOST"          , INT2FIX(CU_CTX_MAP_HOST));
    rb_define_const(rb_cCUContextFlagsEnum, "LMEM_RESIZE_TO_MAX", INT2FIX(CU_CTX_LMEM_RESIZE_TO_MAX));

    rb_cCULimitEnum = rb_define_class_under(rb_mCU, "CULimitEnum", rb_cObject);
    rb_define_const(rb_cCULimitEnum, "STACK_SIZE"      , INT2FIX(CU_LIMIT_STACK_SIZE));
    rb_define_const(rb_cCULimitEnum, "PRINTF_FIFO_SIZE", INT2FIX(CU_LIMIT_PRINTF_FIFO_SIZE));

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
    rb_define_method(rb_cCUFunction, "launch_grid"    , (VALUE(*)(ANYARGS))function_launch_grid    , -1);

    rb_cCUStream = rb_define_class_under(rb_mCU, "CUStream", rb_cObject);
    rb_define_alloc_func(rb_cCUStream, stream_alloc);
    rb_define_method(rb_cCUStream, "initialize" , (VALUE(*)(ANYARGS))stream_initialize , 0);
    rb_define_method(rb_cCUStream, "create"     , (VALUE(*)(ANYARGS))stream_create     , 1);
    rb_define_method(rb_cCUStream, "destroy"    , (VALUE(*)(ANYARGS))stream_destroy    , 0);
    rb_define_method(rb_cCUStream, "query"      , (VALUE(*)(ANYARGS))stream_query      , 0);
    rb_define_method(rb_cCUStream, "synchronize", (VALUE(*)(ANYARGS))stream_synchronize, 0);

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

    cuInit(0);
}
