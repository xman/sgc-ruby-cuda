#
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
#

module SGC
module Helper

# @param [Class] klass The name of the class to obtain.
# @return [String] The class name of the class without module string.
#
# @example Obtain the class name of the class.
#     module A
#         module B
#             class C
#             end
#         end
#     end
#     classname(A::B::C)    #=> "C"
def classname(klass)
    klass.name.gsub(/.*\:\:/, "")
end
module_function :classname

end # module
end # module
