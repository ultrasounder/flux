"""
Flux - GPU-native EM field explorer for SI/PI insight
Phase 1: Minimal rendercanvas backend (no wgpu.gui dependency)
"""

import numpy as np
import h5py
import pygfx as gfx
from rendercanvas import RenderCanvas


class SyntheticFieldGenerator:
    @staticmethod
    def dipole_field(shape=(256, 256, 256), dipole_pos=(0.5, 0.5, 0.5)):
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
    def plane_wave_field(shape=(256, 256, 256), wavelength=4):
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


class FluxViewerCanvas(RenderCanvas):
    def __init__(self):
        super().__init__(title="Flux - EM Field Inspector", size=(1200, 800))
        
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(50, 1.5)
        self.camera.position.z = 1.5
        
        self.volume = None
        self.threshold = 0.2
        
        # Load initial field
        self.load_dipole()
    
    def render(self):
        with self.render_context() as frame:
            frame.clear(0.05, 0.05, 0.05, 1.0)
            frame.render(self.scene, self.camera)
    
    def render_field(self, field_data, title):
        if self.volume and self.volume in self.scene.children:
            self.scene.remove(self.volume)
        
        field_norm = field_data / (field_data.max() + 1e-8)
        texture = gfx.Texture(field_norm[np.newaxis, :, :, :], dim=3)
        material = gfx.VolumeSliceMaterial(clim=(self.threshold, 1.0))
        self.volume = gfx.Volume(texture, material)
        self.scene.add(self.volume)
        
        print(f"✓ {title} | Shape: {field_data.shape} | Max: {field_data.max():.3f}")
    
    def load_dipole(self):
        print("Generating dipole field...")
        field = SyntheticFieldGenerator.dipole_field(shape=(256, 256, 256))
        self.render_field(field, "Dipole Field (256³)")
    
    def load_plane_wave(self):
        print("Generating plane wave...")
        field = SyntheticFieldGenerator.plane_wave_field(shape=(256, 256, 256))
        self.render_field(field, "Plane Wave (256³)")
    
    def update_threshold(self, delta):
        self.threshold = np.clip(self.threshold + delta, 0.0, 1.0)
        if self.volume:
            self.volume.material.clim = (self.threshold, 1.0)
        print(f"Threshold: {self.threshold:.1%}")


if __name__ == '__main__':
    canvas = FluxViewerCanvas()
    
    print("\n=== Flux Phase 1 ===")
    print("Controls: [D]ipole [P]lane [+/-] threshold [-Q] quit\n")
    
    # Keyboard handler
    @canvas.add_event_handler('key_down')
    def on_key(event):
        if event.key.name in ('Escape', 'q'):
            canvas.close()
        elif event.key.name == 'd':
            canvas.load_dipole()
        elif event.key.name == 'p':
            canvas.load_plane_wave()
        elif event.key.name in ('Plus', 'Equal'):
            canvas.update_threshold(0.05)
        elif event.key.name == 'Minus':
            canvas.update_threshold(-0.05)
    
    canvas.show()