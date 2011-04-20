require 'ffi'


module FFI

# This class is obtained from ffi-tk (https://github.com/Tass/ffi-tk).
class PrettyStruct < FFI::Struct
    ACCESSOR_CODE = <<-CODE
        def {name}; self[{sym}]; end
        def {name}=(value) self[{sym}] = value; end
    CODE

    def self.layout(*kvs)
        kvs.each_slice(2) do |key, value|
            eval ACCESSOR_CODE.gsub(/\{(.*?)\}/, '{name}' => key, '{sym}' => ":#{key}")
        end

        super
    end

    def members
        layout.members
    end

    def inspect
        kvs = members.zip(values)
        kvs.map!{|key, value| "%s=%s" % [key, value.inspect] }
        "<%s %s>" % [self.class, kvs.join(' ')]
    end
end

end # module
