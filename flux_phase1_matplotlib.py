"""
Flux - EM Field Explorer Phase 1
Minimal MVP using matplotlib 3D for rendering
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class SyntheticFieldGenerator:
    @staticmethod
    def dipole_field(shape=(64, 64, 64), dipole_pos=(0.5, 0.5, 0.5)):
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
    def plane_wave_field(shape=(64, 64, 64), wavelength=4):
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


class FluxViewer:
    """Matplotlib-based 3D field explorer."""
    
    def __init__(self):
        self.field_data = None
        self.threshold = 0.2
        self.field_title = ""
    
    def render_field(self, field_data, title, threshold=0.2):
        """Render orthogonal slices + isosurface of field."""
        self.field_data = field_data
        self.field_title = title
        self.threshold = threshold
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"{title} | Shape: {field_data.shape} | Threshold: {threshold:.1%}", 
                     fontsize=14, fontweight='bold')
        
        # Get middle slices
        mid_x = field_data.shape[0] // 2
        mid_y = field_data.shape[1] // 2
        mid_z = field_data.shape[2] // 2
        
        # XY slice (z = middle)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(field_data[:, :, mid_z], cmap='inferno', origin='lower')
        ax1.set_title(f"XY Slice (z={mid_z})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(ax1.images[0], ax=ax1, label="E-field")
        
        # XZ slice (y = middle)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(field_data[:, mid_y, :], cmap='inferno', origin='lower')
        ax2.set_title(f"XZ Slice (y={mid_y})")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        plt.colorbar(ax2.images[0], ax=ax2, label="E-field")
        
        # YZ slice (x = middle)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(field_data[mid_x, :, :], cmap='inferno', origin='lower')
        ax3.set_title(f"YZ Slice (x={mid_x})")
        ax3.set_xlabel("Y")
        ax3.set_ylabel("Z")
        plt.colorbar(ax3.images[0], ax=ax3, label="E-field")
        
        # 3D scatter of high-field voxels (thresholded)
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        high_field = field_data > threshold
        coords = np.argwhere(high_field)
        
        if len(coords) > 0:
            # Subsample for visualization
            step = max(1, len(coords) // 5000)
            coords_sub = coords[::step]
            values_sub = field_data[high_field][::step]
            
            scatter = ax4.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
                                 c=values_sub, cmap='inferno', s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax4, label="E-field", pad=0.1)
        
        ax4.set_title(f"3D Field (threshold > {threshold:.1%})")
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")
        
        # Field statistics
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis('off')
        stats_text = f"""
        Field Statistics
        ─────────────────
        Shape:     {field_data.shape}
        Min:       {field_data.min():.4f}
        Max:       {field_data.max():.4f}
        Mean:      {field_data.mean():.4f}
        Median:    {np.median(field_data):.4f}
        Std Dev:   {field_data.std():.4f}
        
        Voxels > threshold: {np.sum(high_field)}
        Percentage:  {100*np.sum(high_field)/field_data.size:.1f}%
        """
        ax5.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
                verticalalignment='center', transform=ax5.transAxes)
        
        # Energy density (E^2)
        ax6 = fig.add_subplot(2, 3, 6)
        energy = field_data ** 2
        ax6.hist(energy.flatten(), bins=100, edgecolor='black')
        ax6.set_xlabel("Energy Density (E²)")
        ax6.set_ylabel("Voxel Count")
        ax6.set_title("Energy Distribution")
        ax6.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n✓ Rendered {title}")
        print(f"  Shape: {field_data.shape}")
        print(f"  Threshold: {threshold:.1%}")
        print(f"  High-field voxels: {np.sum(high_field)} ({100*np.sum(high_field)/field_data.size:.1f}%)")


def main():
    viewer = FluxViewer()
    
    print("=" * 60)
    print("Flux Phase 1 - EM Field Viewer")
    print("=" * 60)
    
    # Load dipole
    print("\n[1/3] Generating dipole field...")
    dipole = SyntheticFieldGenerator.dipole_field(shape=(64, 64, 64))
    viewer.render_field(dipole, "Dipole Field", threshold=0.15)
    
    # Save to HDF5
    print("\n[2/3] Saving to HDF5...")
    FieldDataStore.save_field('/tmp/dipole_field.h5', dipole,
                             origin=(0, 0, 0),
                             spacing=(1e-6, 1e-6, 1e-6),
                             units='m')
    print("  ✓ Saved to /tmp/dipole_field.h5")
    
    # Load from HDF5
    print("\n[3/3] Loading from HDF5...")
    loaded, metadata = FieldDataStore.load_field('/tmp/dipole_field.h5')
    print(f"  ✓ Loaded {metadata['shape']}")
    print(f"  Origin: {metadata['origin']}")
    print(f"  Spacing: {metadata['spacing']} {metadata['units']}")
    
    # Also demo plane wave
    print("\n[4/4] Rendering plane wave...")
    plane_wave = SyntheticFieldGenerator.plane_wave_field(shape=(64, 64, 64))
    viewer.render_field(plane_wave, "Plane Wave Field", threshold=0.25)


if __name__ == '__main__':
    main()