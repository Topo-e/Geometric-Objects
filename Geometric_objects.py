import bpy
import math
import numpy as np


bl_info = {
    "name": "Geometric objects",
    "author": "Alexis B",
    "verison": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Geometry > Geometric objects", 
    "description": "Create geometric objects like manifolds or other surfaces",
    "category": "Add Mesh",
}


def create_mesh_object(context, verts, edges, faces, name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()

    from bpy_extras import object_utils
    return object_utils.object_data_add(context, mesh, operator=None)


def execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type):
    verts = []
    faces = []

    uStep = (range_u_max - range_u_min) / range_u_step
    vStep = (range_v_max - range_v_min) / range_v_step

    uRange = range_u_step + 1
    vRange = range_v_step + 1

    if wrap_u:
        uRange = uRange - 1

    if wrap_v:
        vRange = vRange - 1

    for vN in range(vRange):
        v = range_v_min + (vN * vStep)

        for uN in range(uRange):
            u = range_u_min + (uN * uStep)

            x = x_func(u, v)
            y = y_func(u, v)
            z = z_func(u, v)

            verts.append((x, y, z))

    for vN in range(range_v_step):
        vNext = vN + 1

        if wrap_v and (vNext >= vRange):
            vNext = 0

        for uN in range(range_u_step):
            uNext = uN + 1

            if wrap_u and (uNext >= uRange):
                uNext = 0

            faces.append([(vNext * uRange) + uNext,
                          (vNext * uRange) + uN,
                          (vN * uRange) + uN,
                          (vN * uRange) + uNext])

    if close_v and wrap_u and (not wrap_v):
        for uN in range(1, range_u_step - 1):
            faces.append([
                range_u_step - 1,
                range_u_step - 1 - uN,
                range_u_step - 2 - uN])
            faces.append([
                range_v_step * uRange,
                range_v_step * uRange + uN,
                range_v_step * uRange + uN + 1])

    if not verts:
        return {'CANCELLED'}

    if surface_type == 1:
        obj = create_mesh_object(context, verts, [], faces, "Surface")
        return {'FINISHED'}
    else:
        return verts, [], faces


# Boy's surface

def boys_surface(context, a, b,
                 range_u_min=0, range_u_max=1, range_u_step=64, wrap_u=False,
                 range_v_min=0, range_v_max=2*math.pi, range_v_step=128, wrap_v=True,
                 close_v=False):
    def z(u, v):
        return np.exp(1j * v) * u

    def g1(u, v):
        z_val = z(u, v)
        k = (z_val * (1 - z_val ** 4)) / (z_val ** 6 + math.sqrt(5) * z_val ** 3 - 1)
        return -a * k.imag

    def g2(u, v):
        z_val = z(u, v)
        k = (z_val * (1 + z_val ** 4)) / (z_val ** 6 + math.sqrt(5) * z_val ** 3 - 1)
        return -a * k.real

    def g3(u, v):
        z_val = z(u, v)
        k = (1 + z_val ** 6) / (z_val ** 6 + math.sqrt(5) * z_val ** 3 - 1)
        return k.imag - b
    
    def g(u,v):
        return g1(u, v) ** 2 + g2(u, v) ** 2 + g3(u, v) ** 2

    def x_func(u, v):
        return g1(u, v) / g(u,v)

    def y_func(u, v):
        z_val = z(u, v)
        return g2(u, v) / g(u,v)

    def z_func(u, v):
        z_val = z(u, v)
        return -g3(u, v) / g(u,v)

    return execute_surface(context, x_func, y_func, z_func,
                           range_u_min, range_u_max, range_u_step, wrap_u,
                           range_v_min, range_v_max, range_v_step, wrap_v,
                           close_v, surface_type=1)
    
class BoysSurface(bpy.types.Operator) :
    bl_idname = "object.boys_surface"
    bl_label = "Boy's surface"
    bl_description = "Create a Boy's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            min = 0.7,
            max = 3
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.7,
            max = 3
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 10,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 10,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        
        boys_surface(bpy.context,a=self.a, b=self.b, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 3/2
        self.b = 0.63
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)

    
# Breather's surface

def breather_surface(context, b,
                     range_u_min=-13.2, range_u_max=13.2, range_u_step=60, wrap_u=False,
                     range_v_min=-37.4, range_v_max=37.4, range_v_step=150, wrap_v=False,
                     close_v=False):
    # Constantes
    r = 1.0 - b ** 2
    w = np.sqrt(r)

    # Définition de la fonction denom
    def denom(u, v):
        return b * ((w * math.cosh(b * u)) ** 2 + (b * math.sin(w * v)) ** 2) 

    # Définition des fonctions x, y et z
    def x_func(u, v):
        d = denom(u, v)
        x = -u + (2 * r * math.cosh(b * u) * math.sinh(b * u)) / d 
        return x * 0.4 

    def y_func(u, v):
        d = denom(u, v)
        y = (2 * w * math.cosh(b * u) * (-(w * math.cos(v) * math.cos(w * v)) - math.sin(v) * math.sin(w * v))) / d 
        return y * 0.4

    def z_func(u, v):
        d = denom(u, v)
        z = z = (2 * w * math.cosh(b * u) * (-(w * math.sin(v) * math.cos(w * v)) + math.cos(v) * math.sin(w * v))) / d 
        return z * 0.4

    return execute_surface(context, x_func, y_func, z_func,
                           range_u_min, range_u_max, range_u_step, wrap_u,
                           range_v_min, range_v_max, range_v_step, wrap_v,
                           close_v, surface_type=1)

class BreatherSurface(bpy.types.Operator) :
    bl_idname = "object.breather_surface"
    bl_label = "Breather surface"
    bl_description = "Create a Breather's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.1,
            max = 3
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 10,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 10,
            max = 256
    )
  
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        breather_surface(bpy.context, b = self.b, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.b = 0.4
        self.range_u_step = 60
        self.range_v_step = 150

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Calabi-Yau manifold
    
def calabi_yau_surface(context, n1, n2, phi, 
                       range_u_min=-1, range_u_max=1, range_u_step=20, wrap_u=False,
                       range_v_min=0, range_v_max=math.pi / 2, range_v_step=64, wrap_v=False,
                       close_v=False):
                                     
    def z01(k1):
        return np.exp(1j * (2 * np.pi * k1) / n1) 
    
    def z02(k2):
        return np.exp(1j * (2 * np.pi * k2) / n2) 
    
    def z1(u, v, k1):
        return ((np.cosh(u + 1j * v)) ** (2.0 / n1))
    
    def z2(u, v, k2):
        return ((np.sinh(u + 1j * v)) ** (2.0 / n2))

    def x_func(u, v, k1, k2):
        return (z01(k1) * z1(u, v, k1)).real

    def y_func(u, v, k1, k2):
        return (z02(k2) * z2(u,v, k2)).real

    def z_func(u, v, k1, k2):
        return (np.cos(phi) * (z01(k1) * z1(u, v, k1)).imag) + (np.sin(phi) * (z02(k2) * z2(u, v, k2)).imag)

    # List to store mesh data
    meshes_data = []

    # Integration of loops in the calabi_yau_surface function
    for k1 in range(n1):
        for k2 in range(n2):
            verts, _, faces = execute_surface(context, 
                                              lambda u, v: x_func(u, v, k1, k2), 
                                              lambda u, v: y_func(u, v, k1, k2), 
                                              lambda u, v: z_func(u, v, k1, k2),
                                              range_u_min, range_u_max, range_u_step, wrap_u,
                                              range_v_min, range_v_max, range_v_step, wrap_v,
                                              close_v, surface_type=2)
            meshes_data.append((verts, faces))

    # Create a single mesh containing all mesh data
    vertices_combined = []
    faces_combined = []

    for verts, faces in meshes_data:
        vertices_offset = len(vertices_combined)
        vertices_combined.extend(verts)
        faces_combined.extend([(v_idx + vertices_offset) for v_idx in face] for face in faces)

    mesh_combined = bpy.data.meshes.new("Calabi_Yau_Main_Mesh")
    mesh_combined.from_pydata(vertices_combined, [], faces_combined)
    mesh_combined.update()

    # Create an object from the combined mesh
    main_object = bpy.data.objects.new("Calabi_Yau_Main_Object", mesh_combined)
    context.collection.objects.link(main_object)

    return main_object

class CalabiYauSurface(bpy.types.Operator):
    bl_idname = "object.calabi_yau_surface"
    bl_label = "Calabi-Yau manifold"
    bl_description = "Create a Calabi-Yau manifold"
    bl_options = {"REGISTER", "UNDO"}
    
    n1 : bpy.props.IntProperty (
            name = "n1",
            description = "This is a size parameter",
            min = 1,
            max = 10
    )
    
    n2 : bpy.props.IntProperty (
            name = "n2",
            description = "This is a size parameter",
            min = 1,
            max = 10
    )
    
    phi : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.1,
            max = 3
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 10,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 10,
            max = 256
    )
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        calabi_yau_surface(context, n1=self.n1, n2=self.n2, phi=self.phi, 
                           range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.n1 = 2
        self.n2 = 2
        self.phi = np.pi / 4
        self.range_u_step = 20
        self.range_v_step = 64

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Dupin cyclide

def dupin_surface(context, a, b,
                 range_u_min=-np.pi, range_u_max=np.pi, range_u_step=128, wrap_u=False,
                 range_v_min= -np.pi, range_v_max=np.pi, range_v_step=256, wrap_v=False,
                 close_v=False):
    
    c = np.sqrt(a**2 - b**2)
    d = c
    
    def x_func(u, v):
        return ((d * (c - (a * np.cos(u) * np.cos(v)))) + (b**2 * np.cos(u))) / (a - (c * np.cos(u) * np.cos(v))) * 0.2

    def y_func(u, v):
        return (b * np.sin(u) * (a - (d * np.cos(v)))) / (a - (c * np.cos(u) * np.cos(v))) * 0.2

    def z_func(u, v):
        return (b * np.sin(v) * ((c * np.cos(u)) - d)) / (a - (c * np.cos(u) * np.cos(v))) * 0.2

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class DupinSurface(bpy.types.Operator) :
    bl_idname = "object.dupin_surface"
    bl_label = "Dupin cyclide"
    bl_description = "Create a Dupin cyclide"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            default = 4,
            min = 0.1,
            max = 10
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            default = 3,
            min = 1,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            default = 64,
            min = 8,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            default = 128,
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        dupin_surface(bpy.context, a=self.a, b=self.b, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 4
        self.b = 3
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Two-sheeted Hyperboloid

def hyperboloid_2(context, a, b, c,
                 range_u_min=-4, range_u_max=4, range_u_step=64, wrap_u=False,
                 range_v_min=0, range_v_max= 2 * np.pi, range_v_step=128, wrap_v=False,
                 close_v=False):
    
    def x_func(u, v):
        return a * np.sqrt(u**2 - 1) * np.cos(v)

    def y_func(u, v):
        return b * np.sqrt(u**2 - 1) * np.sin(v)

    def z_func(u, v):
        return c * u
    
    return execute_surface(context, x_func, y_func, z_func,
                           range_u_min, range_u_max, range_u_step, wrap_u,
                           range_v_min, range_v_max, range_v_step, wrap_v,
                           close_v, surface_type=1)

class Hyperboloid2(bpy.types.Operator) :
    bl_idname = "object.hyperboloid_2"
    bl_label = "Hyperboloid two-sheeted"
    bl_description = "Create a Hyperboloid with two sheeted"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    c : bpy.props.FloatProperty (
            name = "c",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    range_u_min : bpy.props.FloatProperty (
            name = "u_min",
            min = -15,
            max = 0
    )
    
    range_u_max : bpy.props.FloatProperty (
            name = "u_max",
            min = 0,
            max = 15
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
    
    range_v_min : bpy.props.FloatProperty (
            name = "v_min",
            min = 0,
            max = 2 * np.pi
    )
    
    range_v_max : bpy.props.FloatProperty (
            name = "v_max",
            min = 0,
            max = 2 * np.pi
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        hyperboloid_2(bpy.context, a=self.a, b=self.b, c=self.c, range_u_min=self.range_u_min, range_u_max=self.range_u_max, range_v_min=self.range_v_min, range_v_max=self.range_v_max,range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 0.4
        self.b = 0.4
        self.c = 0.4
        self.range_u_min = -4
        self.range_u_max = 4
        self.range_v_min = 0
        self.range_v_max = 2 * np.pi
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Jeener's flower

def jeener_flower(context, a, b,
                 range_u_min=-2, range_u_max=2, range_u_step=32, wrap_u=False,
                 range_v_min=-2, range_v_max= 2, range_v_step=64, wrap_v=False,
                 close_v=False):
    
    def x_func(u, v):
        w = u + 1j * v
        return np.real((w**3) / 3 - (w**5) / 5) / b

    def y_func(u, v):
        w = u + 1j * v
        return np.real(1j * ((w**3) / 3 + (w**5) / 5)) / b

    def z_func(u, v):
        w = u + 1j * v
        return np.real((w**4) / a) /b
    
    return execute_surface(context, x_func, y_func, z_func,
                           range_u_min, range_u_max, range_u_step, wrap_u,
                           range_v_min, range_v_max, range_v_step, wrap_v,
                           close_v, surface_type=1)

class JeenerFlower(bpy.types.Operator) :
    bl_idname = "object.jeener_flower"
    bl_label = "Jeener's flower"
    bl_description = "Create a Jeener's flower"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            default = 2,
            min = 0.1,
            max = 10
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            default = -9,
            min = -40,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            default = 32,
            min = 8,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            default = 64,
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        jeener_flower(bpy.context, a=self.a, b=self.b, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 2
        self.b = -17
        self.range_u_step = 32
        self.range_v_step = 64

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Klein's surface

def klein_surface(context, a, b, c,
                  range_u_min=0, range_u_max=2 * np.pi, range_u_step=64, wrap_u=False,
                  range_v_min=0, range_v_max=2 * np.pi, range_v_step=128, wrap_v=False,
                  close_v=False):
    
    def ru(u):
        return c * (1 - np.cos(u) / 2)
    
    def x_func(u, v):
        return (a * (1 + np.sin(u)) + ru(u) * np.cos(v)) * np.cos(u) if u <= np.pi else (a * (1 + np.sin(u)) * np.cos(u) - ru(u) * np.cos(v))

    def y_func(u, v):
        return (b + ru(u) * np.cos(v)) * np.sin(u) if u <= np.pi else b * np.sin(u)

    def z_func(u, v):
        return ru(u) * np.sin(v)

    vertices = []
    edges = []
    faces = []
    
    u_values = np.linspace(range_u_min, range_u_max, range_u_step)
    for u in u_values:
        for v in np.linspace(range_v_min, range_v_max, range_v_step):
            vertices.append((x_func(u, v), y_func(u, v), z_func(u, v)))
    
    if not close_v:
        for i in range(range_u_step):
            for j in range(range_v_step - 1):
                edges.append((i * range_v_step + j, i * range_v_step + j + 1))
            edges.append((i * range_v_step + range_v_step - 1, i * range_v_step))

    # Combine all vertices, edges, and faces
    mesh_data = {
        "vertices": vertices,
        "edges": edges,
        "faces": faces
    }

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class KleinSurface(bpy.types.Operator) :
    bl_idname = "object.klein_surface"
    bl_label = "Klein's surface"
    bl_description = "Create a Klein's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            min = 0,
            max = 10
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0,
            max = 15
    )
    
    c : bpy.props.FloatProperty (
            name = "c",
            description = "This is a size parameter",
            min = 0,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        klein_surface(bpy.context, a=self.a, b=self.b, c=self.c, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 0.5
        self.b = 1.45
        self.c = 0.65
        self.range_u_step = 32
        self.range_v_step = 64

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Kuen's surface

def kuen_surface(context, 
                 range_u_min=-2.46* np.pi, range_u_max=2.46 * np.pi, range_u_step=64, wrap_u=True,
                 range_v_min= -2 * np.pi, range_v_max=2 * np.pi, range_v_step=128, wrap_v=False, 
                 close_v=True):
    
    def x_func(u, v):
        return (2 * np.cosh(v) * (np.cos(u) + u * np.sin(u)) / (np.cosh(v) * np.cosh(v) + u * u)) * 0.5
    
    def y_func(u, v):
        return (2 * np.cosh(v) * (-u * np.cos(u) + np.sin(u)) / (np.cosh(v) * np.cosh(v) + u * u)) *0.5
    
    def z_func(u, v):
        return (v - (2 * np.sinh(v) * np.cosh(v)) / (np.cosh(v) * np.cosh(v) + u * u)) * 0.5

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class KuenSurface(bpy.types.Operator) :
    bl_idname = "object.kuen_surface"
    bl_label = "Kuen's surface"
    bl_description = "Create a Kuen's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        kuen_surface(bpy.context, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Morin's surface

def morin_surface(context, n, k , a,
                 range_u_min=0, range_u_max=np.pi, range_u_step=64, wrap_u=False,
                 range_v_min= 0, range_v_max=2*np.pi, range_v_step=128, wrap_v=False,
                 close_v=False):
    
    def x_func(u, v):
        K1 = np.cos(u) / (np.sqrt(2) - k * np.sin(2 * u) * np.sin(n * v))
        return (K1 * ((2.0 / (n - 1)) * np.cos(u) * np.cos((n - 1) * v) + np.sqrt(2) * np.sin(u) * np.cos(v))) * 0.5

    def y_func(u, v):
        K1 = np.cos(u) / (np.sqrt(2) - k * np.sin(2 * u) * np.sin(n * v))
        return (K1 * ((2.0 / (n - 1)) * np.cos(u) * np.sin((n - 1) * v) - np.sqrt(2) * np.sin(u) * np.sin(v))) * 0.5
    
    def z_func(u, v):
        K1 = np.cos(u) / (np.sqrt(2) - k * np.sin(2 * u) * np.sin(n * v))
        return a * K1 * np.cos(u) * 0.5

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class MorinSurface(bpy.types.Operator) :
    bl_idname = "object.morin_surface"
    bl_label = "Morin's surface"
    bl_description = "Create a Morin's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    n : bpy.props.IntProperty (
            name = "a",
            description = "This is a size parameter",
            min = 2,
            max = 10
    )
    
    k : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.1,
            max = 15
    )
    
    a : bpy.props.FloatProperty (
            name = "c",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        morin_surface(bpy.context, n=self.n, k=self.k, a=self.a, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.n = 2
        self.k = 1
        self.a = 2
        self.range_u_step = 32
        self.range_v_step = 64

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Oloid

class Oloid(bpy.types.Operator) :
    bl_idname = "object.oloid"
    bl_label = "Oloid"
    bl_description = "Create an Oloid"
    bl_options = {"REGISTER", "UNDO"}
    
    #space : bpy.props.FloatProperty (name="espace", description="Space between differents cubes ")
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        bpy.ops.mesh.primitive_circle_add(vertices=64, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode":1}, TRANSFORM_OT_translate={"value":(1, 0, 0), "orient_axis_ortho":'X', "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(True, False, False),  "use_snap_edit":True, "use_snap_nonedit":True})
        bpy.ops.transform.rotate(value=1.5708, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.convex_hull()
        bpy.ops.object.editmode_toggle()

        return {'FINISHED'}


# Plucker's surface

def plucker_surface(context, a, n,
                 range_u_min=-20, range_u_max=20, range_u_step=128, wrap_u=False,
                 range_v_min= -0.5*np.pi, range_v_max=0.5*np.pi, range_v_step=256, wrap_v=False,
                 close_v=False):
    def x_func(u, v):
        return u * np.cos(v)

    def y_func(u, v):
        return u * np.sin(v)

    def z_func(u, v):
        return a * np.cos(n * v)

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class PluckerSurface(bpy.types.Operator) :
    bl_idname = "object.plucker_surface"
    bl_label = "Plucker's surface"
    bl_description = "Create a Plucker's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            min = 0.1,
            max = 5
    )
    
    n : bpy.props.IntProperty (
            name = "n",
            description = "This is a size parameter",
            min = 0,
            max = 17
    )
    
    range_u_min : bpy.props.FloatProperty (
            name = "u_min",
            min = -10,
            max = 0
    )
    
    range_u_max : bpy.props.FloatProperty (
            name = "u_max",
            min = 0,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
        
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        plucker_surface(bpy.context, a=self.a, n=self.n, range_u_min=self.range_u_min, range_u_max=self.range_u_max,range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 0.5
        self.n = 4
        self.range_u_min = -1.5
        self.range_u_max = 1.5
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Richmond's surface

def richmond_surface(context, n,
                 range_u_min=0.53, range_u_max=1.1, range_u_step=24, wrap_u=False,
                 range_v_min= 0, range_v_max=2*np.pi, range_v_step=128, wrap_v=False,
                 close_v=False):
    
    
    def x_func(u, v):
        z = u * np.exp(1j * v)
        return (-0.5  / z - (z ** (2 * n + 1)) / ((4 * n + 2))).real * 1.5

    def y_func(u, v):
        z = u * np.exp(1j * v)
        return ((-0.5 * 1j) / z + (1j * z ** (2 * n+1)) / ((4 * n + 2))).real * 1.5
    
    def z_func(u, v):
        z = u * np.exp(1j * v)
        return (z**n).real / n * 1.5

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class RichmondSurface(bpy.types.Operator) :
    bl_idname = "object.richmond_surface"
    bl_label = "Richmond's surface"
    bl_description = "Create a Richmond's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    n : bpy.props.IntProperty (
            name = "n",
            description = "This is a size parameter",
            min = 0,
            max = 17
    )
    
    range_u_min : bpy.props.FloatProperty (
            name = "u_min",
            min = 0.25,
            max = 1.2
    )
    
    range_u_max : bpy.props.FloatProperty (
            name = "u_max",
            min = 0.35,
            max = 1.3
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
        
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        richmond_surface(bpy.context, n=self.n, range_u_min=self.range_u_min, range_u_max=self.range_u_max,range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.n = 7
        self.range_u_min = 0.5
        self.range_u_max = 1.1
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Rose flower

def rose_flower(context, 
                 range_u_min=1.0e-6, range_u_max=1.0, range_u_step=33, wrap_u=False,
                 range_v_min=-20 * math.pi / 9, range_v_max=15 * math.pi, range_v_step=500, wrap_v=False,
                 close_v=False):
    
    #theta1 = -20 * math.pi / 9
    #theta2 = 15 * math.pi
    #x0 = 0.7831546645625248
    
    def diam(theta):
        return 0.5 * math.pi * math.exp(-theta / (8 * math.pi))
    
    def theta_new(theta):
        return theta
    
    def y1(x1, theta):
        return 1.9565284531299512 * x1**2 * (1.2768869870150188 * x1 - 1.0)**2 * math.sin(diam(theta))
    
    def x(theta):
        return 1.0 - 0.5 * ((1.25 * (1 - (3.6 * theta % (2 * math.pi)) / math.pi)**2) - 0.25)
    
    def r(x1, theta):
        return x(theta) * (x1 * math.sin(diam(theta)) + y1(x1, theta) * math.cos(diam(theta)))
      
    def x_func(x1, theta):
        return r(x1, theta) * math.sin(theta_new(theta))
    
    def y_func(x1, theta):
        return r(x1, theta) * math.cos(theta_new(theta))
    
    def z_func(x1, theta):
        return x(theta) * (x1 * math.cos(diam(theta)) - y1(x1, theta) * math.sin(diam(theta)))
    
    return execute_surface(context, x_func, y_func, z_func,
                           range_u_min, range_u_max, range_u_step, wrap_u,
                           range_v_min, range_v_max, range_v_step, wrap_v,
                           close_v, surface_type=1)

class RoseFlower(bpy.types.Operator) :
    bl_idname = "object.rose_flower"
    bl_label = "Rose flower"
    bl_description = "Create a Rose flower"
    bl_options = {"REGISTER", "UNDO"}
    
    range_v_min : bpy.props.IntProperty (
            name = "v_min",
            min = -10,
            max = 5
    )
    
    range_v_max : bpy.props.IntProperty (
            name = "v_max",
            min = -4,
            max = 5
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 66
    )
        
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 1000
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        rose_flower(bpy.context, range_v_min=self.range_v_min*-20*math.pi/9, range_v_max=self.range_v_max*15*math.pi/9,range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.range_v_min = -8
        self.range_v_max = -1
        self.range_u_step = 33
        self.range_v_step = 500

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Scherk-Collins surface

def scherk_collins_surface(context, nstory, ntwist, rwarp, wflange,
                            range_u_min=0, range_u_max=None, range_u_step=150, wrap_u=False,
                            range_v_min=1.0e-6, range_v_max=None, range_v_step=10, wrap_v=False,
                            close_v=False):
    range_u_max = 0.5 * nstory * np.pi
    range_v_max = wflange / 2
        
    # Fonction Twist
    def Twist(p, theta):
        x = p[0] * np.cos(theta) - p[1] * np.sin(theta)
        y = p[0] * np.sin(theta) + p[1] * np.cos(theta)
        z = p[2]
        return [x, y, z]

    # Fonction Warp
    def Warp(p, theta):
        r = p[0] + rwarp
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = p[1]
        return [x, y, z]

    # Fonction Scherk
    def Scherk(u, v, xsign, ysign):
        z_val = u + 1j * v
        t1 = np.sqrt(2 * 1/np.tan(z_val))
        t2 = 1 + 1/np.tan(z_val)
        x = (0.5 * xsign * (np.real(np.log(t1 - t2)) - np.real(np.log(t1 + t2)))) / np.sqrt(2.0)
        y = (ysign * np.real(1j * (np.arctan(1 - t1) - np.arctan(1 + t1)))) / np.sqrt(2.0)
        z = np.real(z_val)
        return [x, y, z]

    # x_func, y_func and z_func functions defined at the global level
    def x_func(u, v, xsign, ysign):
        theta = 4 * u / nstory 
        x, _, _ = Warp(Twist(Scherk(u, v, xsign, ysign), ntwist * theta), theta)
        return x * 0.3

    def y_func(u, v, xsign, ysign):
        theta = 4 * u / nstory 
        _, y, _ = Warp(Twist(Scherk(u, v, xsign, ysign), ntwist * theta), theta)
        return y * 0.3

    def z_func(u, v, xsign, ysign):
        theta = 4 * u / nstory 
        _, _, z = Warp(Twist(Scherk(u, v, xsign, ysign), ntwist * theta), theta)
        return z * 0.3

    # List to store mesh data
    meshes_data = []

    # Loops on xsign and ysign
    for v_sign in [-1, 1]:
        for u_sign in [-1, 1]:
            verts, _, faces = execute_surface(context, 
                                              lambda u, v: x_func(u, v, u_sign, v_sign), 
                                              lambda u, v: y_func(u, v, u_sign, v_sign), 
                                              lambda u, v: z_func(u, v, u_sign, v_sign),
                                              range_u_min, range_u_max, range_u_step, wrap_u,
                                              range_v_min, range_v_max, range_v_step, wrap_v,
                                              close_v, surface_type=2)
            meshes_data.append((verts, faces))

    # Créer un seul maillage contenant toutes les données de maillage
    vertices_combined = []
    faces_combined = []

    for verts, faces in meshes_data:
        vertices_offset = len(vertices_combined)
        vertices_combined.extend(verts)
        faces_combined.extend([(v_idx + vertices_offset) for v_idx in face] for face in faces)

    mesh_combined = bpy.data.meshes.new("Scherk_Collins_Main_Mesh")
    mesh_combined.from_pydata(vertices_combined, [], faces_combined)
    mesh_combined.update()

    # Create a single mesh containing all mesh data
    main_object = bpy.data.objects.new("Scherk_Collins_Main_Object", mesh_combined)
    context.collection.objects.link(main_object)

    return main_object

class ScherkCollinsSurface(bpy.types.Operator) :
    bl_idname = "object.scherk_collins_surface"
    bl_label = "Scherk-Collins surface"
    bl_description = "Create a Scherk-Collins surface"
    bl_options = {"REGISTER", "UNDO"}
    
    nstory : bpy.props.IntProperty (
            name = "n story",
            description = "This is a size parameter",
            min = 1,
            max = 17
    )
    
    ntwist : bpy.props.IntProperty (
            name = "n twist",
            description = "This is a size parameter",
            min = 0,
            max = 17
    )
    
    rwarp : bpy.props.FloatProperty (
            name = "r warp",
            min = 0,
            max = 10
    )
    
    wflange : bpy.props.FloatProperty (
            name = "w flange",
            min = 0.1,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 5,
            max = 300
    )
        
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 2,
            max = 30
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        scherk_collins_surface(bpy.context, nstory=self.nstory, ntwist=self.ntwist, rwarp=self.rwarp, wflange=self.wflange, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.nstory = 10
        self.ntwist = 1
        self.rwarp = 4
        self.wflange = 2.25
        self.range_u_step = 150
        self.range_v_step = 10

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


# Soucoupoid

def soucoupoide_surface(context, a, b , c,
                 range_u_min=-np.pi, range_u_max=np.pi, range_u_step=64, wrap_u=True,
                 range_v_min= 0, range_v_max=2*np.pi, range_v_step=128, wrap_v=True,
                 close_v=False):
    
    def x_func(u, v):
        return a * np.cos(u) * np.cos(v)

    def y_func(u, v):
        return b * np.cos(u) * np.sin(v)
    
    def z_func(u, v):
        return c * (np.sin(u)**3 )

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class Soucoupoid(bpy.types.Operator) :
    bl_idname = "object.soucoupoide_surface"
    bl_label = "Soucoupoid"
    bl_description = "Create a Soucoupoid's surface"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    c : bpy.props.FloatProperty (
            name = "c",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 10,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 10,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        soucoupoide_surface(bpy.context, a=self.a , b=self.b , c=self.c, range_u_step=self.range_u_step, range_v_step=self.range_v_step)
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 1.5
        self.b = 1.5
        self.c = 0.75
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)

    
# Tannery pear

def tannery_surface(context, a, b , c, 
                 range_u_min=-np.pi, range_u_max=np.pi, range_u_step=64, wrap_u=True,
                 range_v_min= 0, range_v_max=2*np.pi, range_v_step=128, wrap_v=True,
                 close_v=False):
    
    def x_func(u, v):
        return a * np.sin(2 * u) * np.cos(v)

    def y_func(u, v):
        return b * np.sin(2 * u) * np.sin(v)
    
    def z_func(u, v):
        return c * np.sin(u)

    return execute_surface(context, x_func, y_func, z_func,
                    range_u_min, range_u_max, range_u_step, wrap_u,
                    range_v_min, range_v_max, range_v_step, wrap_v,
                    close_v, surface_type=1)

class TanneryPear(bpy.types.Operator) :
    bl_idname = "object.tannery_surface"
    bl_label = "Tannery pear"
    bl_description = "Create a Tannery pear"
    bl_options = {"REGISTER", "UNDO"}
    
    a : bpy.props.FloatProperty (
            name = "a",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    b : bpy.props.FloatProperty (
            name = "b",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    c : bpy.props.FloatProperty (
            name = "c",
            description = "This is a size parameter",
            min = 0.1,
            max = 10
    )
    
    range_u_step : bpy.props.IntProperty (
            name = "Step u",
            description = "This is the number of step in u direction",
            min = 8,
            max = 128
    )
    
    range_v_step : bpy.props.IntProperty (
            name = "Step v",
            description = "This is the number of step in v direction",
            min = 8,
            max = 256
    )
    
    @classmethod
    def poll(cls, context) :
        return True
    
    def execute(self, context) :
        tannery_surface(bpy.context, a=self.a , b=self.b , c=self.c, range_u_step=self.range_u_step, range_v_step=self.range_v_step ) 
        return {'FINISHED'}
    
    def reset_properties(self):
        # Resetting properties to their default values
        self.a = 0.65
        self.b = 0.65
        self.c = 1.25
        self.range_u_step = 64
        self.range_v_step = 128

    def invoke(self, context, event):
        self.reset_properties()
        return self.execute(context)


class MyPanel(bpy.types.Panel):
    bl_idname = "VIEW_3D_PT_manifold"
    bl_label = "Geometric objects"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Geometric objects"
    bl_description = "Some geometric objects like manifolds or other surfaces"
    
    def draw_header(self, context):
        self.layout.label(icon="LIGHTPROBE_CUBEMAP")
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        c.operator(BoysSurface.bl_idname)
        c.operator(BreatherSurface.bl_idname)
        c.operator(CalabiYauSurface.bl_idname)
        c.operator(DupinSurface.bl_idname)
        c.operator(Hyperboloid2.bl_idname)
        c.operator(JeenerFlower.bl_idname)
        c.operator(KleinSurface.bl_idname)
        c.operator(KuenSurface.bl_idname)
        c.operator(MorinSurface.bl_idname)
        c.operator(Oloid.bl_idname)
        c.operator(PluckerSurface.bl_idname)
        c.operator(RichmondSurface.bl_idname)
        c.operator(RoseFlower.bl_idname)
        c.operator(ScherkCollinsSurface.bl_idname)
        c.operator(Soucoupoid.bl_idname)
        c.operator(TanneryPear.bl_idname)
        
            
def register():
    bpy.utils.register_class(BoysSurface)
    bpy.utils.register_class(BreatherSurface)
    bpy.utils.register_class(CalabiYauSurface)
    bpy.utils.register_class(DupinSurface)
    bpy.utils.register_class(Hyperboloid2)
    bpy.utils.register_class(JeenerFlower)
    bpy.utils.register_class(KleinSurface)
    bpy.utils.register_class(KuenSurface)
    bpy.utils.register_class(MorinSurface)
    bpy.utils.register_class(Oloid)
    bpy.utils.register_class(PluckerSurface)
    bpy.utils.register_class(RichmondSurface)
    bpy.utils.register_class(RoseFlower)
    bpy.utils.register_class(ScherkCollinsSurface)
    bpy.utils.register_class(Soucoupoid)
    bpy.utils.register_class(TanneryPear)
    bpy.utils.register_class(MyPanel)
    
def unregister():
    bpy.utils.unregister_class(BoysSurface)
    bpy.utils.unregister_class(BreatherSurface)
    bpy.utils.unregister_class(CalabiYauSurface)
    bpy.utils.unregister_class(DupinSurface)
    bpy.utils.unregister_class(Hyperboloid2)
    bpy.utils.unregister_class(JeenerFlower)
    bpy.utils.unregister_class(KleinSurface)
    bpy.utils.unregister_class(KluenSurface)
    bpy.utils.unregister_class(MorinSurface)
    bpy.utils.unregister_class(Oloid)
    bpy.utils.unregister_class(PluckerSurface)
    bpy.utils.unregister_class(RichmondSurface)
    bpy.utils.unregister_class(RoseFlower)
    bpy.utils.unregister_class(ScherkCollinsSurface)
    bpy.utils.unregister_class(Soucoupoid)
    bpy.utils.unregister_class(TanneryPear)
    bpy.utils.unregister_class(MyPanel)
    

if __name__ == "__main__":
    register()
    