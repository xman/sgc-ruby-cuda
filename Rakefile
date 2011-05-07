require 'rubygems'
require 'rake/gempackagetask'
require 'rake/testtask'
require 'rake/clean'
require 'yard'

load 'version.rb'

CUDA_PATH         = "lib/cuda"
CUDA_DRIVER_PATH  = "#{CUDA_PATH}/driver"
CUDA_RUNTIME_PATH = "#{CUDA_PATH}/runtime"
DOC_PATH          = "doc"
HTML_OUTPUT_PATH  = "html"


task :default => []

desc 'Build everything.'
task :all => [:package, :yard]


spec = Gem::Specification.new do |s|
    s.platform    = Gem::Platform::RUBY
    s.name        = 'sgc-ruby-cuda'
    s.version     = SGC_RUBY_CUDA_VERSION
    s.summary     = 'Ruby bindings for using Nvidia CUDA.'
    s.description = 'SGC-Ruby-CUDA implements Ruby bindings to Nvidia CUDA SDK. It provides easy access to CUDA-enabled GPU from a Ruby program.'

    s.required_ruby_version     = '>= 1.9.2'

    s.author            = 'Chung Shin Yee'
    s.email             = 'shinyee@speedgocomputing.com'
    s.homepage          = 'https://rubyforge.org/projects/rubycuda'
    s.rubyforge_project = 'rubycuda'

    s.require_path = 'lib'

    s.files  = FileList['lib/**/*.rb', "#{DOC_PATH}/**/*.rdoc"].to_a
    s.files += ['Rakefile', 'version.rb', 'README.rdoc', 'COPYING']
    s.files += ['ChangeLog.txt', 'RELEASE.txt']
    s.files += ['.yardopts']
    s.test_files = FileList['test/{**/*.rb,vadd.cu,bad.ptx}'].to_a

    s.add_dependency 'ffi', '>= 1.0.7'
    s.add_dependency 'yard', '>= 0.6.7'

    s.requirements << 'CUDA Toolkit 4.0'
    s.requirements << 'C++ compiler'
    s.requirements << 'CUDA-enabled GPU'
end

Rake::GemPackageTask.new(spec) do |pkg|
    pkg.need_tar_gz  = true
end


desc 'Generate SGC Ruby CUDA documentation with YARD.'
task :yard
YARD::Rake::YardocTask.new do |t|
    t.files = FileList['lib/**/*.rb'].to_a
    t.options += ['-o', "#{HTML_OUTPUT_PATH}"]
end


desc 'Run SGC Ruby CUDA test cases.'
task :test
Rake::TestTask.new do |t|
    t.libs << 'lib'

    t.test_files = FileList['test/**/test_*.rb'].to_a
    t.verbose    = true
end


CLEAN.include ['pkg', "#{HTML_OUTPUT_PATH}"]
CLEAN.include ['**/*.o', '**/*.so']
CLEAN.include ['test/vadd.ptx']
