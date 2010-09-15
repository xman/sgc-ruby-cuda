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

#include "ruby.h"
#include "cuda.h"

// {{{ SGC Ruby modules.
static VALUE rb_mSGC;
static VALUE rb_mCU;
// }}}

// {{{ CUDA Ruby classes.
static VALUE rb_cCUDevice;
static VALUE rb_cCUContext;
static VALUE rb_cCUModule;
static VALUE rb_cCUFunction;
static VALUE rb_cCUDevicePtr;
// }}}

// {{{ SGC Ruby classes.
static VALUE rb_cInt32Buffer;
static VALUE rb_cFloat32Buffer;
// }}}

// {{{ SGC C/C++ structures.
template <typename TElement>
struct TypedBuffer {
    size_t size;
    TElement* p;
};

typedef struct TypedBuffer<int>   Int32Buffer;
typedef struct TypedBuffer<float> Float32Buffer;
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

static VALUE device_compute_capability(VALUE self)
{
    CUdevice* p;
    Data_Get_Struct(self, CUdevice, p);
    int major;
    int minor;
    cuDeviceComputeCapability(&major, &minor, *p);
    return rb_ary_new3(2, INT2FIX(major), INT2FIX(minor));
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

static VALUE module_get_function(VALUE self, VALUE str)
{
    CUmodule* p;
    Data_Get_Struct(self, CUmodule, p);
    CUfunction* pfunc = new CUfunction;
    cuModuleGetFunction(pfunc, *p, StringValuePtr(str));
    return Data_Wrap_Struct(rb_cCUFunction, 0, generic_free<CUfunction>, pfunc);
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


// {{{ Buffer
template <typename TElement>
static void buffer_free(void* p)
{
    TElement* pbuffer = static_cast<TElement*>(p);
    delete[] pbuffer->p;
    delete pbuffer;
}

template <typename TElement>
static VALUE buffer_alloc(VALUE klass)
{
    TElement* p = new TElement;
    return Data_Wrap_Struct(klass, 0, buffer_free<TElement>, p);
}

template <typename TElement>
static VALUE buffer_initialize(VALUE self, VALUE nelements)
{
    typedef struct TypedBuffer<TElement> TBuffer;

    size_t n = NUM2ULONG(nelements);
    TBuffer* pbuffer;
    Data_Get_Struct(self, TBuffer, pbuffer);
    pbuffer->size = nelements;
    pbuffer->p = new TElement[n];
    for (size_t i = 0; i < n; ++i) {
        pbuffer->p[i] = TElement(0);
    }
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
    TElement element = pbuffer->p[i];
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
    pbuffer->p[i] = v;
    return value;
}
typedef VALUE (*BufferElementSetFunctionType)(VALUE, VALUE, VALUE);
// }}}


// {{{ Memory transfer functions.
static VALUE memcpy_htod(VALUE self, VALUE rb_device_ptr, VALUE rb_buffer, VALUE rb_nbytes)
{
    CUdeviceptr* pdevice_ptr;
    Int32Buffer* pbuffer;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_buffer, Int32Buffer, pbuffer);
    size_t nbytes = NUM2ULONG(rb_nbytes);
    cuMemcpyHtoD(*pdevice_ptr, pbuffer->p, nbytes);
    return Qnil; // TODO: Return the status of the transfer.
}

static VALUE memcpy_dtoh(VALUE self, VALUE rb_buffer, VALUE rb_device_ptr, VALUE rb_nbytes)
{
    Int32Buffer* pbuffer;
    CUdeviceptr* pdevice_ptr;
    Data_Get_Struct(rb_device_ptr, CUdeviceptr, pdevice_ptr);
    Data_Get_Struct(rb_buffer, Int32Buffer, pbuffer);
    size_t nbytes = NUM2ULONG(rb_nbytes);
    cuMemcpyDtoH(pbuffer->p, *pdevice_ptr, nbytes);
    return Qnil; // TODO: Return the status of the transfer.
}
// }}}


extern "C" void Init_rubycu()
{
    rb_mSGC = rb_define_module("SGC");
    rb_mCU  = rb_define_module_under(rb_mSGC, "CU");

    rb_cCUDevice = rb_define_class_under(rb_mCU, "CUDevice", rb_cObject);
    rb_define_alloc_func(rb_cCUDevice, device_alloc);
    rb_define_method(rb_cCUDevice, "initialize", (VALUE(*)(ANYARGS))device_initialize, -1);
    rb_define_method(rb_cCUDevice, "get"       , (VALUE(*)(ANYARGS))device_get       ,  1);
    rb_define_method(rb_cCUDevice, "compute_capability", (VALUE(*)(ANYARGS))device_compute_capability, 0);

    rb_cCUContext = rb_define_class_under(rb_mCU, "CUContext", rb_cObject);
    rb_define_alloc_func(rb_cCUContext, context_alloc);
    rb_define_method(rb_cCUContext, "initialize", (VALUE(*)(ANYARGS))context_initialize, -1);
    rb_define_method(rb_cCUContext, "create"    , (VALUE(*)(ANYARGS))context_create    ,  2);
    rb_define_method(rb_cCUContext, "destroy"   , (VALUE(*)(ANYARGS))context_destroy   ,  0);

    rb_cCUModule = rb_define_class_under(rb_mCU, "CUModule", rb_cObject);
    rb_define_alloc_func(rb_cCUModule, module_alloc);
    rb_define_method(rb_cCUModule, "initialize"  , (VALUE(*)(ANYARGS))module_initialize  , -1);
    rb_define_method(rb_cCUModule, "load"        , (VALUE(*)(ANYARGS))module_load        ,  1);
    rb_define_method(rb_cCUModule, "get_function", (VALUE(*)(ANYARGS))module_get_function,  1);

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

    rb_cInt32Buffer = rb_define_class_under(rb_mCU, "Int32Buffer", rb_cObject);
    rb_define_alloc_func(rb_cInt32Buffer, buffer_alloc<Int32Buffer>);
    rb_define_method(rb_cInt32Buffer, "initialize", (VALUE(*)(ANYARGS))static_cast<BufferInitializeFunctionType>(&buffer_initialize<int>) , 1);
    rb_define_method(rb_cInt32Buffer, "[]"        , (VALUE(*)(ANYARGS))static_cast<BufferElementGetFunctionType>(&buffer_element_get<int>), 1);
    rb_define_method(rb_cInt32Buffer, "[]="       , (VALUE(*)(ANYARGS))static_cast<BufferElementSetFunctionType>(&buffer_element_set<int>), 2);

    rb_cFloat32Buffer = rb_define_class_under(rb_mCU, "Float32Buffer", rb_cObject);
    rb_define_alloc_func(rb_cFloat32Buffer, buffer_alloc<Float32Buffer>);
    rb_define_method(rb_cFloat32Buffer, "initialize", (VALUE(*)(ANYARGS))static_cast<BufferInitializeFunctionType>(&buffer_initialize<float>) , 1);
    rb_define_method(rb_cFloat32Buffer, "[]"        , (VALUE(*)(ANYARGS))static_cast<BufferElementGetFunctionType>(&buffer_element_get<float>), 1);
    rb_define_method(rb_cFloat32Buffer, "[]="       , (VALUE(*)(ANYARGS))static_cast<BufferElementSetFunctionType>(&buffer_element_set<float>), 2);

    rb_define_module_function(rb_mCU, "memcpy_htod", (VALUE(*)(ANYARGS))memcpy_htod, 3);
    rb_define_module_function(rb_mCU, "memcpy_dtoh", (VALUE(*)(ANYARGS))memcpy_dtoh, 3);

    cuInit(0);
}
