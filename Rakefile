require 'rubygems'
require 'rake/gempackagetask'
require 'rake/rdoctask'
require 'rake/testtask'
require 'rake/clean'

CUDA_PATH         = "lib/cuda"
CUDA_DRIVER_PATH  = "#{CUDA_PATH}/driver"
CUDA_RUNTIME_PATH = "#{CUDA_PATH}/runtime"
RUBYCU_LIB        = "#{CUDA_DRIVER_PATH}/rubycu.so"
RUBYCU_LIB_DEP    = ["#{CUDA_DRIVER_PATH}/extconf.rb", "#{CUDA_DRIVER_PATH}/rubycu.cpp"]


task :default => [:build]

desc 'Build everything.'
task :all => [:build, :package, :rdoc]

desc 'Build all SGC Ruby CUDA libraries.'
task :build => [:rubycu]


desc 'Build rubycu shared library.'
task :rubycu => RUBYCU_LIB

file RUBYCU_LIB => RUBYCU_LIB_DEP do
    system %{cd #{CUDA_DRIVER_PATH}; ruby extconf.rb; make}
end


spec = Gem::Specification.new do |s|
    s.platform    = Gem::Platform::CURRENT
    s.name        = 'sgc-ruby-cuda'
    s.version     = File.read('VERSION').strip
    s.summary     = 'Ruby bindings for using Nvidia CUDA.'
    s.description = 'SGC-Ruby-CUDA implements Ruby bindings to Nvidia CUDA SDK. It provides easy access to CUDA-enabled GPU from a Ruby program.'

    s.required_ruby_version     = '>= 1.9.2'
    s.required_rubygems_version = '>= 1.3.6'

    s.author            = 'Chung Shin Yee'
    s.email             = 'shinyee@speedgocomputing.com'
    s.homepage          = 'https://rubyforge.org/projects/rubycuda'
    s.rubyforge_project = 'rubycuda'

    s.require_path = 'lib'

    s.files      = FileList['lib/**/*.rb', RUBYCU_LIB].to_a
    s.files     += ['Rakefile', 'VERSION', 'COPYING']
    s.files.reject! { |f| f.include? 'extconf.rb' }
    s.test_files = FileList['test/{**/test_*.rb,vadd.cu,bad.ptx}'].to_a

    s.has_rdoc         = true
    s.extra_rdoc_files = ['README.rdoc']

    s.requirements << 'CUDA Toolkit 3.1'
    s.requirements << 'C++ compiler'
    s.requirements << 'CUDA-enabled GPU'
end

Rake::GemPackageTask.new(spec) do |pkg|
    pkg.need_tar_gz  = true
end


desc 'Generate SGC Ruby CUDA documentation.'
Rake::RDocTask.new do |r|
    r.main       = 'README.rdoc'

    r.rdoc_files.include 'README.rdoc'
    r.rdoc_files.include 'lib/**/*.rb', 'lib/**/*.cpp'

    r.options << '--inline-source'
    r.options << '--line-numbers'
    r.options << '--all'
    r.options << '--fileboxes'
    r.options << '--diagram'
end


Rake::TestTask.new do |t|
    t.libs << 'lib'

    t.test_files = FileList['test/**/test_*.rb']
    t.verbose    = true
end


CLEAN.include ['pkg', 'html']
CLEAN.include ["#{CUDA_DRIVER_PATH}/{Makefile,mkmf.log}"]
CLEAN.include ['**/*.o', '**/*.so']
CLEAN.include ['test/vadd.ptx']
