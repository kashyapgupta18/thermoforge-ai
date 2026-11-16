"""
THERMOFORGE AI - 3D Surface Geometry Extractor for WebGL / Three.js
Extracts boundary quad faces and triangular surface mesh with interpolated node temperatures
from the 3D hexahedral finite element mesh.
"""

import numpy as np
from typing import Dict, List, Any
try:
    from .thermal_engine import HexahedralMesh
except ImportError:
    from thermal_engine import HexahedralMesh

def extract_surface_mesh_for_threejs(mesh: HexahedralMesh, temperatures_celsius: List[float]) -> Dict[str, Any]:
    """
    Extracts all exterior boundary faces from active hexahedral elements,
    converts them into triangles (positions, normals, temperature values per vertex),
    and formats them for Three.js BufferGeometry.
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    T_arr = np.array(temperatures_celsius)

    # Face definitions for an element:
    # 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
    face_nodes = [
        [0, 4, 7, 3],  # -X face
        [1, 2, 6, 5],  # +X face
        [0, 1, 5, 4],  # -Y face
        [3, 7, 6, 2],  # +Y face
        [0, 3, 2, 1],  # -Z face
        [4, 5, 6, 7]   # +Z face
    ]

    face_normals = [
        [-1,  0,  0],
        [ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  1,  0],
        [ 0,  0, -1],
        [ 0,  0,  1]
    ]

    neighbors = [
        (-1,  0,  0),
        ( 1,  0,  0),
        ( 0, -1,  0),
        ( 0,  1,  0),
        ( 0,  0, -1),
        ( 0,  0,  1)
    ]

    tri_positions = []
    tri_normals = []
    tri_temperatures = []
    wireframe_lines = []

    # Center coordinates around origin for easy Three.js orbit rotation
    cx = mesh.lx / 2.0
    cy = mesh.ly / 2.0
    cz = mesh.lz / 2.0

    nodes_centered = mesh.nodes.copy()
    nodes_centered[:, 0] -= cx
    nodes_centered[:, 1] -= cy
    nodes_centered[:, 2] -= cz

    # Scale to convenient viewing size in Three.js (e.g. max dimension ~ 5 units)
    scale_factor = 5.0 / max(mesh.lx, mesh.ly, mesh.lz)
    nodes_centered *= scale_factor

    # Track heat source nodes for hotspot marker
    heat_source_coords = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mesh.active_elements[i, j, k]:
                    continue

                elem_nodes = mesh.get_element_nodes(i, j, k)
                if mesh.heat_source_elements[i, j, k]:
                    # Centroid of this die element
                    centroid = np.mean(nodes_centered[elem_nodes], axis=0)
                    heat_source_coords.append(centroid.tolist())

                for f_idx, (di, dj, dk) in enumerate(neighbors):
                    ni, nj, nk = i + di, j + dj, k + dk
                    is_boundary = False
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                        is_boundary = True
                    elif not mesh.active_elements[ni, nj, nk]:
                        is_boundary = True

                    if is_boundary:
                        fn = elem_nodes[face_nodes[f_idx]]
                        n_vec = face_normals[f_idx]

                        # Two triangles per quad face: (0, 1, 2) and (0, 2, 3)
                        # Triangle 1
                        for idx in [fn[0], fn[1], fn[2]]:
                            tri_positions.extend(nodes_centered[idx].tolist())
                            tri_normals.extend(n_vec)
                            tri_temperatures.append(float(T_arr[idx]))

                        # Triangle 2
                        for idx in [fn[0], fn[2], fn[3]]:
                            tri_positions.extend(nodes_centered[idx].tolist())
                            tri_normals.extend(n_vec)
                            tri_temperatures.append(float(T_arr[idx]))

                        # Wireframe edges
                        wireframe_lines.extend(nodes_centered[fn[0]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[1]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[1]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[2]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[2]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[3]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[3]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[0]].tolist())

    return {
        "positions": tri_positions,
        "normals": tri_normals,
        "temperatures": tri_temperatures,
        "wireframe_lines": wireframe_lines,
        "num_triangles": len(tri_positions) // 9,
        "scale_factor": scale_factor,
        "heat_source_centroids": heat_source_coords
    }
