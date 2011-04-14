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

class Flags

    # @param [IEnum] e An enum that implements {IEnum} interface.
    # @param [Symbol, Integer, Array<Symbol, Integer>] *flags The list of flags to include in the returning value.
    # @raise [ArgumentError] Invalid symbol or value.
    # @return [Integer] The composite value of the _flags_ with respect to _e_.
    #
    # @example Compute the composite value of flags with multiple symbols and integers.
    #     CUEventFlags.symbols            #=> [:DEFAULT, :BLOCKING_SYNC, :DISABLE_TIMING]
    #     CUEventFlags[:DEFAULT]          #=> 0
    #     CUEventFlags[0]                 #=> :DEFAULT
    #     CUEventFlags[:BLOCKING_SYNC]    #=> 1
    #     CUEventFlags[1]                 #=> :BLOCKING_SYNC
    #     Flags.value(CUEventFlags, :DISABLE_TIMING)                      #=> 2
    #     Flags.value(CUEventFlags, :BLOCKING_SYNC, :DISABLE_TIMING)      #=> 3
    #     Flags.value(CUEventFlags, [:BLOCKING_SYNC, :DISABLE_TIMING])    #=> 3
    #     Flags.value(CUEventFlags, [ 1, :DISABLE_TIMING])                #=> 3
    def self.value(e, *flags)
        case flags
            when Array
                f = 0
                flags.flatten.each do |x|
                    case x
                        when Symbol
                            v = e[x] or raise_invalid_symbol(e, x)
                            f |= v
                        when Integer
                            e[x] or raise_invalid_value(e, x)
                            f |= x
                        else
                            raise ArgumentError, "Invalid flags: #{x.to_s}. Expect Symbol or Integer in the flags array."
                    end
                end
                f
            else raise ArgumentError, "Invalid flags: #{flags.to_s}. Expect Array<Symbol, Integer>, Symbol or Integer."
        end
    end


    # @private
    def self.raise_invalid_symbol(e, symbol)
        raise ArgumentError, "Invalid flags symbol: #{symbol.to_s}. Expect symbol in #{e.symbols.to_s}."
    end
    private_class_method :raise_invalid_symbol


    # @private
    def self.raise_invalid_value(e, value)
        raise ArgumentError, "Invalid flags value: #{value.to_s}."
    end
    private_class_method :raise_invalid_value

end

end # module
end # module
