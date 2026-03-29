import pygfx as gfx
import numpy as np
from rendercanvas.auto import RenderCanvas, loop

# 1. Generate Synthetic 3D Field Data
def generate_field_data(shape=(128, 128, 128)):
    x = np.linspace(-5, 5, shape[0])
    y = np.linspace(-5, 5, shape[1])
    z = np.linspace(-5, 5, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    data = np.exp(-(xx**2 + yy**2 + zz**2) / 2.0).astype(np.float32)
    return data

field_data = generate_field_data()

# 2. Setup Canvas and Renderer
canvas = RenderCanvas(size=(1000, 800), title="EM-Insight Phase 1 (Unified)")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# 3. Add Volumetric Field (The "Cloud")
tex = gfx.Texture(field_data, dim=3)
geo = gfx.Geometry(grid=tex)

volume_cloud = gfx.Volume(
    geo,
    gfx.VolumeRayMaterial(clim=(0, 1), map=gfx.cm.magma, interpolation="linear")
)
scene.add(volume_cloud)

# 4. Add the Slicer (Using VolumeSliceMaterial)
# The 'plane' parameter is (a, b, c, d) for the equation ax + by + cz + d = 0
field_slice = gfx.Volume(
    geo,
    gfx.VolumeSliceMaterial(clim=(0, 1), map=gfx.cm.magma, plane=(0, 0, 1, 0))
)
scene.add(field_slice)

# 5. Camera and Controller
camera = gfx.PerspectiveCamera(70, 16/9)
camera.show_object(volume_cloud)
controller = gfx.OrbitController(camera, register_events=renderer)

# 6. Animation Logic: Animating the Plane
state = {"z_offset": 0, "direction": 0.1}

def animate():
    state["z_offset"] += state["direction"]
    if abs(state["z_offset"]) > 50:
        state["direction"] *= -1
    
    # Update the plane's 'd' parameter to move it along the Z-axis
    # Plane: 0x + 0y + 1z + d = 0  => z = -d
    field_slice.material.plane = (0, 0, 1, -state["z_offset"])
    
    renderer.render(scene, camera)
    canvas.request_draw()

if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()