"""
Flux - GPU-native EM field explorer for SI/PI insight
Phase 1 Scaffold: Static volumetric field rendering with threshold control

Structure:
- flux/
  - synthetic_fields.py
  - data_loader.py
  - viewer.py
  - main.py (this file)
"""

import numpy as np
import h5py
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

import pygfx as gfx
import wgpu


# ============================================================================
# PART 1: Synthetic Field Generator
# ============================================================================

class SyntheticFieldGenerator:
    """Generate synthetic 3D EM fields for MVP testing."""
    
    @staticmethod
    def dipole_field(shape=(256, 256, 256), dipole_pos=(0.5, 0.5, 0.5)):
        """
        Generate E-field magnitude from a point dipole.
        
        Args:
            shape: (nx, ny, nz) grid dimensions
            dipole_pos: Normalized position (0-1) of dipole center
        
        Returns:
            E_magnitude: 3D array of field magnitude
        """
        nx, ny, nz = shape
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        
        # Shift dipole to specified position
        dx = dipole_pos[0] * 2 - 1
        dy = dipole_pos[1] * 2 - 1
        dz = dipole_pos[2] * 2 - 1
        
        X, Y, Z = np.meshgrid(x - dx, y - dy, z - dz, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2) + 0.1  # Avoid singularity
        
        # Dipole field: E ~ 1/r^2
        E_magnitude = 1.0 / (R**2 + 0.1)
        E_magnitude = np.clip(E_magnitude, 0, 1)  # Normalize to [0, 1]
        
        return E_magnitude.astype(np.float32)
    
    @staticmethod
    def plane_wave_field(shape=(256, 256, 256), wavelength=4):
        """
        Generate E-field magnitude from a plane wave.
        
        Args:
            shape: Grid dimensions
            wavelength: Wavelength in grid units
        
        Returns:
            E_magnitude: 3D field array
        """
        nx, ny, nz = shape
        x = np.linspace(0, 4*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        z = np.linspace(0, 2*np.pi, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Plane wave propagating in +x, polarized in z
        E_magnitude = np.abs(np.sin(2*np.pi*X / wavelength) * 
                            np.sin(Y) * np.sin(Z))
        
        return (E_magnitude / E_magnitude.max()).astype(np.float32)


# ============================================================================
# PART 2: HDF5 Data Loader & Storage
# ============================================================================

class FieldDataStore:
    """Manage HDF5 storage and loading of 3D field data."""
    
    @staticmethod
    def save_field(filepath, field_data, origin=(0, 0, 0), spacing=(1e-6, 1e-6, 1e-6), units='m'):
        """
        Save 3D field to HDF5 with metadata.
        
        Args:
            filepath: Output .h5 file
            field_data: 3D numpy array
            origin: (x0, y0, z0) in physical space
            spacing: (dx, dy, dz) in physical space
            units: 'm', 'um', 'mil', etc.
        """
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('E_magnitude', data=field_data, compression='gzip')
            f.attrs['origin'] = origin
            f.attrs['spacing'] = spacing
            f.attrs['units'] = units
            f.attrs['shape'] = field_data.shape
    
    @staticmethod
    def load_field(filepath):
        """
        Load 3D field from HDF5.
        
        Returns:
            field_data: 3D array
            metadata: Dict with origin, spacing, units, shape
        """
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
# PART 3: Flux Viewer (pygfx + PySide6)
# ============================================================================

class FluxViewer(QMainWindow):
    """Main EM field visualization window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux - EM Field Inspector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # ===== LEFT: pygfx Canvas =====
        self.canvas = wgpu.WgpuCanvas()
        self.canvas.setMinimumSize(QSize(800, 600))
        layout.addWidget(self.canvas, stretch=1)
        
        # Create pygfx scene
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(50, 1.0)
        self.camera.position.z = 1.5
        
        # ===== RIGHT: Control Panel =====
        control_layout = QVBoxLayout()
        
        # Threshold slider
        control_layout.addWidget(QLabel("Threshold (%)"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(20)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        control_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("20%")
        control_layout.addWidget(self.threshold_label)
        
        # Load button
        load_btn = QPushButton("Generate Synthetic Field")
        load_btn.clicked.connect(self.load_dipole_field)
        control_layout.addWidget(load_btn)
        
        plane_wave_btn = QPushButton("Load Plane Wave")
        plane_wave_btn.clicked.connect(self.load_plane_wave)
        control_layout.addWidget(plane_wave_btn)
        
        # Info label
        self.info_label = QLabel("Ready.")
        self.info_label.setFont(QFont("Courier", 9))
        control_layout.addWidget(self.info_label)
        
        control_layout.addStretch()
        
        panel_widget = QWidget()
        panel_widget.setLayout(control_layout)
        panel_widget.setMaximumWidth(300)
        layout.addWidget(panel_widget)
        
        # Initialize
        self.volume = None
        self.field_data = None
        self.threshold_value = 0.2
    
    def load_dipole_field(self):
        """Generate and load dipole field."""
        print("Generating dipole field...")
        field_data = SyntheticFieldGenerator.dipole_field(shape=(256, 256, 256))
        self.render_field(field_data, "Dipole Field (256³)")
    
    def load_plane_wave(self):
        """Generate and load plane wave field."""
        print("Generating plane wave field...")
        field_data = SyntheticFieldGenerator.plane_wave_field(shape=(256, 256, 256))
        self.render_field(field_data, "Plane Wave (256³)")
    
    def render_field(self, field_data, title):
        """Render 3D field volume."""
        self.field_data = field_data
        
        # Remove old volume
        if self.volume in self.scene.children:
            self.scene.remove(self.volume)
        
        # Create volume with colormap
        # Normalize field data to [0, 1]
        field_norm = field_data / (field_data.max() + 1e-8)
        
        # Create pygfx volume
        self.volume = gfx.Volume(
            gfx.Texture(field_norm[np.newaxis, :, :, :], dim=3),
            gfx.VolumeSliceMaterial(clim=(self.threshold_value, 1.0))
        )
        
        self.scene.add(self.volume)
        
        # Update info
        self.info_label.setText(f"{title}\nShape: {field_data.shape}\nMax: {field_data.max():.3f}")
        print(f"Rendered {title}")
    
    def on_threshold_changed(self):
        """Update threshold on slider change."""
        value = self.threshold_slider.value() / 100.0
        self.threshold_value = value
        self.threshold_label.setText(f"{self.threshold_slider.value()}%")
        
        if self.volume is not None:
            # Update material threshold
            self.volume.material.clim = (value, 1.0)
        
        self.canvas.request_draw()
    
    def animate(self):
        """Render loop."""
        with self.canvas.render_context() as frame:
            frame.clear(0.0, 0.0, 0.0, 1.0)
            frame.render(self.scene, self.camera)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    app = QApplication(sys.argv)
    viewer = FluxViewer()
    viewer.show()
    
    # Animation timer
    from PySide6.QtCore import QTimer
    timer = QTimer()
    timer.timeout.connect(viewer.animate)
    timer.start(16)  # ~60 FPS
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()