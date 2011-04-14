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

# @abstract An Enum interface.
# An enum maps a symbol to a value, and a value to a symbol.
module IEnum

    module ClassMethods

        # @return [Array] The list of valid symbols of this Enum class.
        def symbols; raise NotImplementedError; end

        # @param [Symbol, Object] key A symbol or value to use as a key to map.
        # @return [Symbol, Object] The symbol or value that the _key_ maps to.
        #   * If the _key_ is a symbol, return the corresponding value.
        #   * If the _key_ is a value, return the corresponding symbol.
        #   * Return nil if the _key_ is invalid.
        def [](key); raise NotImplementedError; end

    end

    # @private
    def self.included(base)
        base.extend(ClassMethods)
    end

    # @private
    def self.forward(klass, dest)
        klass.instance_eval %{
            def symbols
                #{dest}.symbols
            end

            def [](*args)
                #{dest}[*args]
            end
        }
    end

end

end # module
end # module
