"""
Flux - GPU-native EM field explorer for SI/PI insight
Phase 1: Using rendercanvas.auto (Gemini's working pattern)
"""

import pygfx as gfx
import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import h5py


class SyntheticFieldGenerator:
    @staticmethod
    def dipole_field(shape=(128, 128, 128), dipole_pos=(0.5, 0.5, 0.5)):
        nx, ny, nz = shape
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        
        dx = dipole_pos[0] * 2 - 1
        dy = dipole_pos[1] * 2 - 1
        dz = dipole_pos[2] * 2 - 1
        
        X, Y, Z = np.meshgrid(x - dx, y - dy, z - dz, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2) + 0.1
        E_magnitude = 1.0 / (R**2 + 0.1)
        return np.clip(E_magnitude, 0, 1).astype(np.float32)
    
    @staticmethod
    def plane_wave_field(shape=(128, 128, 128), wavelength=4):
        nx, ny, nz = shape
        x = np.linspace(0, 4*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        z = np.linspace(0, 2*np.pi, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        E_magnitude = np.abs(np.sin(2*np.pi*X / wavelength) * np.sin(Y) * np.sin(Z))
        return (E_magnitude / E_magnitude.max()).astype(np.float32)


class FieldDataStore:
    @staticmethod
    def save_field(filepath, field_data, origin=(0, 0, 0), spacing=(1e-6, 1e-6, 1e-6), units='m'):
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('E_magnitude', data=field_data, compression='gzip')
            f.attrs['origin'] = origin
            f.attrs['spacing'] = spacing
            f.attrs['units'] = units
            f.attrs['shape'] = field_data.shape
    
    @staticmethod
    def load_field(filepath):
        with h5py.File(filepath, 'r') as f:
            field_data = f['E_magnitude'][:]
            metadata = {
                'origin': tuple(f.attrs['origin']),
                'spacing': tuple(f.attrs['spacing']),
                'units': f.attrs['units'].decode() if isinstance(f.attrs['units'], bytes) else f.attrs['units'],
                'shape': tuple(f.attrs['shape']),
            }
        return field_data, metadata


# ============================================================================
# FLUX VIEWER - Phase 1
# ============================================================================

def main():
    print("=" * 60)
    print("Flux Phase 1 - EM Field Inspector")
    print("=" * 60)
    
    # Generate field
    print("\n[1] Generating dipole field (128³)...")
    field_data = SyntheticFieldGenerator.dipole_field(shape=(128, 128, 128))
    print(f"    ✓ Shape: {field_data.shape} | Min: {field_data.min():.3f} | Max: {field_data.max():.3f}")
    
    # Save to HDF5
    print("[2] Saving to HDF5...")
    FieldDataStore.save_field('/tmp/flux_dipole.h5', field_data)
    print("    ✓ Saved to /tmp/flux_dipole.h5")
    
    # Canvas
    print("[3] Initializing WebGPU canvas...")
    canvas = RenderCanvas(size=(1200, 800), title="Flux - EM Field Inspector")
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    print("    ✓ Canvas ready")
    
    # Normalize field for rendering
    field_norm = field_data / (field_data.max() + 1e-8)
    
    # Texture (3D) → Geometry
    tex = gfx.Texture(field_norm, dim=3)
    geo = gfx.Geometry(grid=tex)
    
    # Volume Ray Material (volumetric cloud)
    print("[4] Creating volumetric renderer...")
    volume = gfx.Volume(
        geo,
        gfx.VolumeRayMaterial(
            clim=(0.15, 1.0),  # Threshold: hide below 15%
            map=gfx.cm.inferno
        )
    )
    scene.add(volume)
    print("    ✓ Volume added to scene")
    
    # Slicer (orthogonal plane through Z)
    print("[5] Creating slicer...")
    slicer = gfx.Volume(
        geo,
        gfx.VolumeSliceMaterial(
            clim=(0.15, 1.0),
            map=gfx.cm.inferno,
            plane=(0, 0, 1, 0)  # Z = 0 initially
        )
    )
    scene.add(slicer)
    print("    ✓ Slicer added")
    
    # Camera
    camera = gfx.PerspectiveCamera(70, 16/9)
    camera.show_object(volume)
    controller = gfx.OrbitController(camera, register_events=renderer)
    
    # Animation state
    state = {
        "z_offset": 0,
        "z_direction": 1,
        "threshold": 0.15,
        "field_type": "dipole"
    }
    
    def animate():
        # Animate slicing plane
        state["z_offset"] += state["z_direction"] * 0.5
        if abs(state["z_offset"]) > 50:
            state["z_direction"] *= -1
        
        # Update slicer plane
        slicer.material.plane = (0, 0, 1, -state["z_offset"])
        
        # Render
        renderer.render(scene, camera)
    
    # Keyboard handler
    @canvas.add_event_handler('key_down')
    def on_key(event):
        key_name = event.get('key', '') if isinstance(event, dict) else event.key.name
        key_name = str(key_name).lower() if key_name else ''
        
        if 'escape' in key_name or key_name == 'q':
            canvas.close()
        elif key_name == 'd':
            print("\n[LOAD] Dipole field...")
            field = SyntheticFieldGenerator.dipole_field(shape=(128, 128, 128))
            field_norm = field / (field.max() + 1e-8)
            tex_new = gfx.Texture(field_norm, dim=3)
            geo_new = gfx.Geometry(grid=tex_new)
            volume.geometry = geo_new
            slicer.geometry = geo_new
            print(f"✓ Dipole loaded (max: {field.max():.3f})")
        elif key_name == 'p':
            print("\n[LOAD] Plane wave field...")
            field = SyntheticFieldGenerator.plane_wave_field(shape=(128, 128, 128))
            field_norm = field / (field.max() + 1e-8)
            tex_new = gfx.Texture(field_norm, dim=3)
            geo_new = gfx.Geometry(grid=tex_new)
            volume.geometry = geo_new
            slicer.geometry = geo_new
            print(f"✓ Plane wave loaded (max: {field.max():.3f})")
        elif '+' in key_name or '=' in key_name:
            state["threshold"] = min(state["threshold"] + 0.05, 0.95)
            volume.material.clim = (state["threshold"], 1.0)
            slicer.material.clim = (state["threshold"], 1.0)
            print(f"Threshold: {state['threshold']:.0%}")
        elif '-' in key_name:
            state["threshold"] = max(state["threshold"] - 0.05, 0.0)
            volume.material.clim = (state["threshold"], 1.0)
            slicer.material.clim = (state["threshold"], 1.0)
            print(f"Threshold: {state['threshold']:.0%}")
    
    print("\n" + "=" * 60)
    print("Controls:")
    print("  [D]ipole       - Load dipole field")
    print("  [P]lane Wave   - Load plane wave field")
    print("  [+/-]          - Adjust threshold")
    print("  [Q]uit         - Exit")
    print("=" * 60 + "\n")
    
    # Run
    canvas.request_draw(animate)
    loop.run()


if __name__ == '__main__':
    main()