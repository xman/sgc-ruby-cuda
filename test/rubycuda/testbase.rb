#-----------------------------------------------------------------------
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
#-----------------------------------------------------------------------

require 'tempfile'
require 'rbconfig'
require 'rubycuda'

include SGC::Cuda
include SGC::Cuda::Error


module CudaTestBase

    def setup
        CudaDevice.current = ENV['DEVID'].to_i
    end

    def teardown
        CudaThread.exit
    end


    def prepare_kernel_lib
        src_file = 'test/vadd.cu'
        lib_file = 'test/libvadd.so'
        if File.exists?(lib_file) == false || File.mtime(src_file) > File.mtime(lib_file)
            nvcc_build_dynamic_library(src_file, lib_file)
        end
        lib_file
    end

    def nvcc_build_dynamic_library(src_path, lib_path)
        case Config::CONFIG['target_os']
            when /darwin/    # Build universal binary for i386 and x86_64 platforms.
                f32 = Tempfile.new("rubycuda_test32.so")
                f64 = Tempfile.new("rubycuda_test64.so")
                f32.close
                f64.close
                system %{nvcc -shared -m32 -Xcompiler -fPIC #{src_path} -o #{f32.path}}
                system %{nvcc -shared -m64 -Xcompiler -fPIC #{src_path} -o #{f64.path}}
                system %{lipo -arch i386 #{f32.path} -arch x86_64 #{f64.path} -create -output #{lib_path}}
            else    # Build default platform binary.
                system %{nvcc -shared -Xcompiler -fPIC #{src_path} -o #{lib_path}}
        end
    end

end
