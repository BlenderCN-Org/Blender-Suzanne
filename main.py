import struct

import ModernGL
from ModernGL.ext import obj

from PIL import Image
from pyrr import Matrix44

ctx = ModernGL.create_standalone_context()

prog = ctx.program([
    ctx.vertex_shader('''
        #version 330

        uniform mat4 Mvp;

        in vec3 in_vert;
        in vec3 in_norm;

        out vec3 v_vert;
        out vec3 v_norm;

        void main() {
            gl_Position = Mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
        }
    '''),
    ctx.fragment_shader('''
        #version 330

        uniform vec3 LightPos;
        uniform vec3 Color;

        in vec3 v_vert;
        in vec3 v_norm;

        out vec4 color;

        void main() {
            float lum = abs(1.0 - acos(dot(normalize(LightPos - v_vert), normalize(v_norm))) * 2.0 / 3.14159265);
            color = vec4(Color * sqrt(lum), 1.0);
        }
    '''),
])


mvp = Matrix44.perspective_projection(45.0, 1.0, 0.1, 1000.0)
mvp *= Matrix44.look_at((4.0, 3.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

prog.uniforms['Mvp'].write(mvp.astype('float32').tobytes())
prog.uniforms['LightPos'].value = (4.0, 3.0, 7.0)
prog.uniforms['Color'].value = (0.5, 0.7, 0.9)

model = obj.Obj.open('data/suzanne.obj')

vbo = ctx.buffer(model.pack('vx vy vz nx ny nz'))
vao = ctx.simple_vertex_array(prog, vbo, ['in_vert', 'in_norm'])

fbo = ctx.framebuffer(ctx.renderbuffer((512, 512)))

fbo.use()
ctx.enable(ModernGL.DEPTH_TEST)
ctx.viewport = (0, 0, 512, 512)
ctx.clear(0.9, 0.9, 0.9)
vao.render()

pixels = fbo.read(components=3, alignment=1)
img = Image.frombytes('RGB', fbo.size, pixels).transpose(Image.FLIP_TOP_BOTTOM)
img.save('suzanne.png')
