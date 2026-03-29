"""
Flux - GPU-native EM field explorer for SI/PI insight
Phase 1 Scaffold: Using GLFW + pygfx (bypasses PySide6 macOS Qt issues)

Run with: pip install pygfx wgpu h5py numpy glfw
"""

import numpy as np
import h5py
from pathlib import Path
import glfw
import wgpu
from wgpu.gui.glfw import WgpuCanvas as _WgpuCanvas
import pygfx as gfx

# Fallback for newer wgpu versions
try:
    from wgpu.gui.glfw import WgpuCanvas
except ImportError:
    from wgpu.gui._glfw import WgpuCanvas


# ============================================================================
# SYNTHETIC FIELD GENERATOR
# ============================================================================

class SyntheticFieldGenerator:
    """Generate synthetic 3D EM fields for MVP testing."""
    
    @staticmethod
    def dipole_field(shape=(256, 256, 256), dipole_pos=(0.5, 0.5, 0.5)):
        """E-field magnitude from a point dipole."""
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
        E_magnitude = np.clip(E_magnitude, 0, 1)
        
        return E_magnitude.astype(np.float32)
    
    @staticmethod
    def plane_wave_field(shape=(256, 256, 256), wavelength=4):
        """E-field magnitude from a plane wave."""
        nx, ny, nz = shape
        x = np.linspace(0, 4*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        z = np.linspace(0, 2*np.pi, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        E_magnitude = np.abs(np.sin(2*np.pi*X / wavelength) * 
                            np.sin(Y) * np.sin(Z))
        
        return (E_magnitude / E_magnitude.max()).astype(np.float32)


# ============================================================================
# DATA STORE
# ============================================================================

class FieldDataStore:
    """HDF5 storage for 3D field data."""
    
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
# GLFW VIEWER
# ============================================================================

class FluxViewerGLFW:
    """Minimal GLFW + pygfx viewer."""
    
    def __init__(self):
        self.canvas = WgpuCanvas(title="Flux - EM Field Inspector", size=(1200, 800))
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(50, self.canvas.get_logical_size()[0] / self.canvas.get_logical_size()[1])
        self.camera.position.z = 1.5
        
        self.volume = None
        self.field_data = None
        self.threshold = 0.2
        self.field_title = ""
        
        # Load initial field
        self.load_dipole()
    
    def load_dipole(self):
        """Generate and load dipole field."""
        print("Generating dipole field (256³)...")
        field_data = SyntheticFieldGenerator.dipole_field(shape=(256, 256, 256))
        self.render_field(field_data, "Dipole Field")
    
    def load_plane_wave(self):
        """Generate and load plane wave field."""
        print("Generating plane wave field (256³)...")
        field_data = SyntheticFieldGenerator.plane_wave_field(shape=(256, 256, 256))
        self.render_field(field_data, "Plane Wave")
    
    def render_field(self, field_data, title):
        """Render volumetric field."""
        self.field_data = field_data
        self.field_title = title
        
        # Remove old volume
        if self.volume is not None and self.volume in self.scene.children:
            self.scene.remove(self.volume)
        
        # Normalize
        field_norm = field_data / (field_data.max() + 1e-8)
        
        # Create volume
        texture = gfx.Texture(field_norm[np.newaxis, :, :, :], dim=3)
        material = gfx.VolumeSliceMaterial(clim=(self.threshold, 1.0))
        self.volume = gfx.Volume(texture, material)
        self.scene.add(self.volume)
        
        print(f"Rendered {title} | Shape: {field_data.shape} | Max: {field_data.max():.3f}")
    
    def update_threshold(self, delta):
        """Adjust threshold."""
        self.threshold = np.clip(self.threshold + delta, 0.0, 1.0)
        if self.volume is not None:
            self.volume.material.clim = (self.threshold, 1.0)
        print(f"Threshold: {self.threshold:.2%}")
    
    def draw(self):
        """Render frame."""
        with self.canvas.render_context() as frame:
            frame.clear(0.05, 0.05, 0.05, 1.0)
            frame.render(self.scene, self.camera)
    
    def run(self):
        """Main loop."""
        print("\n=== Flux Phase 1 ===")
        print("Controls:")
        print("  [D] Load Dipole Field")
        print("  [P] Load Plane Wave")
        print("  [+] Increase Threshold")
        print("  [-] Decrease Threshold")
        print("  [Q] Quit")
        print()
        
        while not self.canvas.is_closed():
            # Handle keyboard
            for event in self.canvas.get_events():
                if event.type == 'key_down':
                    if event.key.name == 'Escape' or event.key.name == 'q':
                        self.canvas.close()
                    elif event.key.name == 'd':
                        self.load_dipole()
                    elif event.key.name == 'p':
                        self.load_plane_wave()
                    elif event.key.name == 'Plus' or event.key.name == 'Equal':
                        self.update_threshold(0.05)
                    elif event.key.name == 'Minus':
                        self.update_threshold(-0.05)
            
            self.draw()


if __name__ == '__main__':
    viewer = FluxViewerGLFW()
    viewer.run()