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

# Provide methods for evaluating the composite value of flags.
# _self_ which _include/extend_ this module should implement {IEnum} interface.
module FlagsValue

    # @param [Symbol, Integer, Array<Symbol, Integer>] *flags The list of flags to include in the returning value.
    # @raise [ArgumentError] Invalid symbol or value.
    # @return [Integer] The composite value of the _flags_ with respect to _self_.
    #
    # @example Compute the composite value of flags with multiple symbols and integers.
    #     CUEventFlags.symbols            #=> [:DEFAULT, :BLOCKING_SYNC, :DISABLE_TIMING]
    #     CUEventFlags[:DEFAULT]          #=> 0
    #     CUEventFlags[0]                 #=> :DEFAULT
    #     CUEventFlags[:BLOCKING_SYNC]    #=> 1
    #     CUEventFlags[1]                 #=> :BLOCKING_SYNC
    #     CUEventFlags.value(:DISABLE_TIMING)                      #=> 2
    #     CUEventFlags.value(:BLOCKING_SYNC, :DISABLE_TIMING)      #=> 3
    #     CUEventFlags.value([:BLOCKING_SYNC, :DISABLE_TIMING])    #=> 3
    #     CUEventFlags.value([1, :DISABLE_TIMING])                 #=> 3
    def value(*flags)
        flags.empty? == false or raise ArgumentError, "No flags is provided. Expect Array<Symbol, Integer>, Symbol or Integer."

        f = 0
        flags.flatten.each do |x|
            case x
                when Symbol
                    v = self[x] or Pvt::raise_invalid_symbol(x)
                    f |= v
                when Integer
                    self[x] or Pvt::raise_invalid_value(x)
                    f |= x
                else
                    raise ArgumentError, "Invalid flags: #{x.to_s}. Expect Symbol or Integer in the flags array."
            end
        end
        f
    end


    module Pvt

        def raise_invalid_symbol(symbol)
            raise ArgumentError, "Invalid flags symbol: #{symbol.to_s}. Expect symbol in #{self.symbols.to_s}."
        end


        def raise_invalid_value(value)
            raise ArgumentError, "Invalid flags value: #{value.to_s}."
        end

    end

end

end # module
end # module
