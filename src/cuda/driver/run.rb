require 'rubycu'

include SGC::CU

SIZE = 10

d = CUDevice.new
d.get(0)

c = CUContext.new
c.create(0, d)

m = CUModule.new
m.load("vadd.ptx")

da = CUDevicePtr.new
da.mem_alloc(4*SIZE)

db = CUDevicePtr.new
db.mem_alloc(4*SIZE)

dc = CUDevicePtr.new
dc.mem_alloc(4*SIZE)

ha = Int32Buffer.new(SIZE)
hb = Int32Buffer.new(SIZE)
hc = Int32Buffer.new(SIZE)
hd = Int32Buffer.new(SIZE)

(0...SIZE).each { |i| ha[i] = i }
(0...SIZE).each { |i| hb[i] = 2 }
(0...SIZE).each { |i| hc[i] = ha[i] + hb[i] }

memcpy_htod(da, ha, 4*SIZE)
memcpy_htod(db, hb, 4*SIZE)
memcpy_htod(dc, hc, 4*SIZE)

f = m.get_function("vadd");
f.set_param(da, db, dc, SIZE)
f.set_block_shape(SIZE)
f.launch_grid(1)

memcpy_dtoh(hd, dc, 4*SIZE)

puts "A\tB\tCPU\tGPU"
(0...SIZE).each { |i| puts "#{ha[i]}\t#{hb[i]}\t#{hc[i]}\t#{hd[i]}" }
puts

ha = Float32Buffer.new(SIZE)
hb = Float32Buffer.new(SIZE)
hc = Float32Buffer.new(SIZE)
hd = Float32Buffer.new(SIZE)

(0...SIZE).each { |i| ha[i] = i }
(0...SIZE).each { |i| hb[i] = 0.5 }
(0...SIZE).each { |i| hc[i] = ha[i] + hb[i] }

memcpy_htod(da, ha, 4*SIZE)
memcpy_htod(db, hb, 4*SIZE)
memcpy_htod(dc, hc, 4*SIZE)

f = m.get_function("vaddf");
f.set_param(da, db, dc, SIZE)
f.set_block_shape(SIZE)
f.launch_grid(1)

memcpy_dtoh(hd, dc, 4*SIZE)

puts "A\tB\tCPU\tGPU"
(0...SIZE).each { |i| puts "#{ha[i]}\t#{hb[i]}\t#{hc[i]}\t#{hd[i]}" }
puts

da.mem_free
db.mem_free
dc.mem_free
