"""
THERMOFORGE AI - 3D Surface Geometry Extractor for WebGL / Three.js
Extracts boundary quad faces and triangular surface mesh with:
- Interpolated vertex node temperatures
- Heat flux vectors (magnitude & direction -k * ∇T) per face
- Localized hotspot centroid coordinates
- Face normals and wireframe edges
"""

import numpy as np
from typing import Dict, List, Any
try:
    from .thermal_engine import HexahedralMesh
except ImportError:
    from thermal_engine import HexahedralMesh

def extract_surface_mesh_for_threejs(
    mesh: HexahedralMesh,
    temperatures_celsius: List[float],
    conductivity: float = 167.0
) -> Dict[str, Any]:
    """
    Extracts all exterior boundary faces from active hexahedral elements,
    converts them into triangles (positions, normals, temperature values per vertex),
    computes face-level heat flux vectors, and formats them for Three.js.
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
    tri_heat_fluxes = []
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

    # Heat flux vectors representation: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, magnitude]
    heat_flux_vectors = []
    all_flux_mags: List[float] = []
    heat_source_coords = []

    # Find global peak hotspot
    hotspot_idx = int(np.argmax(T_arr))
    hotspot_coord = nodes_centered[hotspot_idx].tolist()
    hotspot_peak_temp = float(T_arr[hotspot_idx])

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mesh.active_elements[i, j, k]:
                    continue

                elem_nodes = mesh.get_element_nodes(i, j, k)
                if mesh.heat_source_elements[i, j, k]:
                    centroid = np.mean(nodes_centered[elem_nodes], axis=0)
                    heat_source_coords.append(centroid.tolist())

                # Estimate element-level gradient ∇T = [∂T/∂x, ∂T/∂y, ∂T/∂z]
                # Using simple finite differences between opposed element faces
                t_elem = T_arr[elem_nodes]
                dT_dx = (np.mean(t_elem[[1, 2, 6, 5]]) - np.mean(t_elem[[0, 3, 7, 4]])) / mesh.dx
                dT_dy = (np.mean(t_elem[[2, 3, 6, 7]]) - np.mean(t_elem[[0, 1, 4, 5]])) / mesh.dy
                dT_dz = (np.mean(t_elem[[4, 5, 6, 7]]) - np.mean(t_elem[[0, 1, 2, 3]])) / mesh.dz

                # Fourier's Law: q_flux = -k * ∇T
                qx = -conductivity * dT_dx
                qy = -conductivity * dT_dy
                qz = -conductivity * dT_dz
                q_mag = float(np.sqrt(qx**2 + qy**2 + qz**2))
                all_flux_mags.append(q_mag)

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
                            tri_heat_fluxes.append(q_mag)

                        # Triangle 2
                        for idx in [fn[0], fn[2], fn[3]]:
                            tri_positions.extend(nodes_centered[idx].tolist())
                            tri_normals.extend(n_vec)
                            tri_temperatures.append(float(T_arr[idx]))
                            tri_heat_fluxes.append(q_mag)

                        # Wireframe edges
                        wireframe_lines.extend(nodes_centered[fn[0]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[1]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[1]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[2]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[2]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[3]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[3]].tolist())
                        wireframe_lines.extend(nodes_centered[fn[0]].tolist())

                        # Store sample heat flux vectors on exposed faces (subsampled for clean rendering)
                        if (i + j + k) % 3 == 0:
                            face_center = np.mean(nodes_centered[fn], axis=0)
                            if q_mag > 1e-4:
                                dir_norm = [float(qx / q_mag), float(qy / q_mag), float(qz / q_mag)]
                                heat_flux_vectors.append({
                                    "origin": face_center.tolist(),
                                    "direction": dir_norm,
                                    "magnitude": q_mag
                                })

    # Heat flux magnitude range over ALL active elements (W/m^2), from q = -k ∇T (element-centre gradient)
    flux_mags = all_flux_mags if all_flux_mags else [0.0]
    q_flux_min = float(np.min(flux_mags))
    q_flux_max = float(np.max(flux_mags))
    q_flux_avg = float(np.mean(flux_mags))

    return {
        "positions": tri_positions,
        "normals": tri_normals,
        "temperatures": tri_temperatures,
        "heat_fluxes": tri_heat_fluxes,
        "wireframe_lines": wireframe_lines,
        "num_triangles": len(tri_positions) // 9,
        "scale_factor": scale_factor,
        "heat_source_centroids": heat_source_coords,
        "heat_flux_vectors": heat_flux_vectors,
        "heat_flux_range": {
            "q_min_w_m2": round(q_flux_min, 2),
            "q_max_w_m2": round(q_flux_max, 2),
            "q_avg_w_m2": round(q_flux_avg, 2)
        },
        "hotspot": {
            "coordinate": hotspot_coord,
            "peak_temperature": hotspot_peak_temp
        }
    }
