import os
import time
from cifkit import Cif
import traceback
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from skspatial.objects import Points
from collections import defaultdict
from numpy.linalg import norm
from re import match
from cifkit.coordination.geometry import compute_polyhedron_metrics
from collections import deque

# import matplotlib.pyplot as plt

# def plot_convex_hull(points, central_atom=None):
#     from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#     points = np.array(points)
#     hull = ConvexHull(points)

#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")

#     for simplex in hull.simplices:
#         triangle = points[simplex]
#         poly = Poly3DCollection([triangle], alpha=0.3, edgecolor='k')
#         poly.set_facecolor('skyblue')
#         ax.add_collection3d(poly)

#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="blue", label="neighbors")

#     if central_atom is not None:
#         ax.scatter(*central_atom, color="red", s=100, label="central atom")

#     ax.set_title("Convex Hull of Polyhedron")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

RE_ELEMENTS = {"Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Lu"}

def polygon_name(triangles):
    mapping = {
        1: 'triangle', 
        2: 'quadrilateral', 
        3: 'pentagon', 
        4: 'hexagon', 
        5: 'heptagon', 
        6: 'octagon'
    }
    return mapping.get(triangles, f"{triangles+2}-gon")

def polyhedron_metrics(center, points_wd):
    points = [[p][0][0] for p in points_wd]
    distances = []
    angles = []
    try:
        points = Points(points)
        if points.are_coplanar():
            print("Points are coplanar")
            
        try:
            hull = ConvexHull(points)
            
            # distance from faces
            for simplex in hull.simplices:
                vertices = points[simplex]
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                offset = -np.dot(normal, vertices[0])
                
                distance = abs(normal[0]*center[0] + normal[1]*center[1] + normal[2]*center[2] + offset) / \
                            np.sqrt(np.square(normal).sum())
                distances.append(distance)
            
            for i in range(len(hull.vertices)):
                p1 = points[hull.vertices[i]]
                p2 = points[hull.vertices[(i + 1) % len(hull.vertices)]]
                p3 = points[hull.vertices[(i - 1) % len(hull.vertices)]]

                v1 = p2 - p1
                v2 = p3 - p1

                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angle = np.degrees(angle)

                angles.append(round(float(angle), 2))
        except:
            print('Error while constructing convex-hull')
            print(traceback.format_exc())
    except:
        print('Error while checking co-planarity')
        print(traceback.format_exc())
        
    if distances:
        distances = np.array(distances)
        distances /= distances.min()
        distances = [round(float(d), 4) for d in distances]
         
    return distances, angles

def compute_largest_face_area(coords, hull, angle_threshold_deg=5.0):
    def triangle_area(a, b, c):
        return 0.5 * norm(np.cross(b - a, c - a))

    def unit_normal(a, b, c):
        normal = np.cross(b - a, c - a)
        return normal / norm(normal)

    def triangles_connected(tri1, tri2):
        # Check if triangles share at least two points (edge)
        return len(set(tri1) & set(tri2)) >= 2

    coords = np.array(coords)
    normals = [unit_normal(*coords[simplex]) for simplex in hull.simplices]
    simplices_used = set()
    preliminary_groups = []

    angle_threshold_rad = np.deg2rad(angle_threshold_deg)

    # Initial grouping based on normal vector similarity
    for idx, normal in enumerate(normals):
        if idx in simplices_used:
            continue
        current_group = [idx]
        simplices_used.add(idx)
        for jdx, other_normal in enumerate(normals):
            if jdx not in simplices_used:
                angle = np.arccos(np.clip(np.dot(normal, other_normal), -1.0, 1.0))
                if angle < angle_threshold_rad or np.pi - angle < angle_threshold_rad:
                    current_group.append(jdx)
                    simplices_used.add(jdx)
        preliminary_groups.append(current_group)

    # Further separate groups based on triangle connectivity
    final_groups = []
    for group in preliminary_groups:
        unvisited = set(group)
        while unvisited:
            connected_group = []
            queue = deque([unvisited.pop()])
            while queue:
                current = queue.popleft()
                connected_group.append(current)
                neighbors = set()
                for candidate in unvisited:
                    if triangles_connected(hull.simplices[current], hull.simplices[candidate]):
                        neighbors.add(candidate)
                queue.extend(neighbors)
                unvisited -= neighbors
            final_groups.append(connected_group)

    # Find the largest face area among connected groups
    largest_area = 0.0
    for face_group in final_groups:
        face_area = sum(triangle_area(*coords[hull.simplices[idx]]) for idx in face_group)
        largest_area = max(largest_area, face_area)

    from collections import Counter
    face_counts = Counter(len(group) for group in final_groups)
    face_descriptions = [f"{count}*{polygon_name(num_triangles)}" for num_triangles, count in sorted(face_counts.items(), reverse=True)]
    face_description_str = ', '.join(face_descriptions)
    
    return round(largest_area, 3), len(final_groups), face_description_str

def get_formula(points_wd, CN, site_symbol_map):
    symbols = [s[2] for s in points_wd[:CN]]
    elements = defaultdict(int)
    for s in symbols:
        elements[site_symbol_map[s]] += 1
    elements = [[k, v] for k, v in elements.items()]
    elements = sorted(elements, key=lambda x: x[1], reverse=True)
        
    formula = ""
    for e, c in elements:
        formula += f"{e}{'' if c==1 else c}"
    return formula

def d_by_dmin_CN_wo_center_type(k, v, center, site_symbol_map):
    
    v = sorted(v, key=lambda x: x[1])
    i_same_as_center = min([100, *[i for i in range(len(v)) if site_symbol_map[v[i][0]] == site_symbol_map[k]]])
    points_wd =[[p[3], p[1], p[0]] for p in v]

    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:20]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    inner_gap_index = np.array(ind_gaps[::-1])
    inner_gap_index = inner_gap_index[inner_gap_index <= i_same_as_center]
    outer_CN = int(inner_gap_index[0]) + 1 if len(inner_gap_index) else 0

    return outer_CN
        
def d_by_dmin(k, v, center, site_symbol_map):
    
    points_wd =[[p[3], p[1], p[0]] for p in v]
    inner_CN = None
    
    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:20]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    
    inner_gap_index = np.array(ind_gaps[::-1])
    outer_CN = d_by_dmin_CN_wo_center_type(k, v, center, site_symbol_map)
    inner_gap_index = inner_gap_index - inner_gap_index[0]
    inner_gap_index = inner_gap_index[:5]
    inner_gap_index = inner_gap_index[inner_gap_index < 0]
    inner_gap_index += outer_CN
    inner_gap_index = inner_gap_index[inner_gap_index >= 4]

    if match(r"[A-Za-z]+", k).group() in RE_ELEMENTS:

        if len(inner_gap_index):
            # keep only the lowest/first
            inner_CN = int(inner_gap_index[0])
            lowest_inner_CN = int(inner_gap_index.min())
            distances, angles = polyhedron_metrics(center, points_wd[:outer_CN])
            distances = [p[1] for p in points_wd[:outer_CN]]

            coords = [p[0] for p in points_wd[:outer_CN]]
            coords.append(center)

            hull = ConvexHull([p[0] for p in points_wd[:outer_CN]])

            # plot_convex_hull([p[0] for p in points_wd[:outer_CN]], central_atom=center)

            metrics = compute_polyhedron_metrics(coords, hull)
            largest_face_area, num_faces, triangles_per_face = compute_largest_face_area(coords, hull)

            return {'site': k, 
                    'outer_CN': outer_CN, 'formula_outer_CN': get_formula(points_wd, outer_CN, site_symbol_map),
                    'inner_CN': inner_CN, 'formula_inner_CN': get_formula(points_wd, inner_CN, site_symbol_map),
                    'lowest_inner_CN': lowest_inner_CN, 'formula_lowest_inner_CN': get_formula(points_wd, lowest_inner_CN, site_symbol_map),
                    'distances': distances, 'angles': angles,'largest_face_area': largest_face_area, 
                    'num_reconstructed_faces': num_faces, 'triangles_per_face': triangles_per_face}, metrics

    return

def find_cenvs_s(cif_path):
    try:
        cif = Cif(cif_path, is_formatted=True)
        # label, symbol, mult, Wyckoff_symbol, x, y, z, occ
        cif.compute_connections()
        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

        conns = cif.connections
        results = []
        data = None
        for i, (k, v) in enumerate(conns.items()):
            if match(r"[A-Za-z]+", k).group() in RE_ELEMENTS:
                wyckoff = f"{loop_vals[2][i]}{loop_vals[3][i]}"
                central_atom = loop_vals[1][i]
                center = v[0][2]
                data, metrics = d_by_dmin(k, v, center, site_symbol_map)
                outer_CN = d_by_dmin_CN_wo_center_type(k, v, center, site_symbol_map)
            
                if data is not None:
                    data['CIF#'] = cif_path.split(os.sep)[-1][:-4]
                    data['central_atom'] = central_atom
                    data['Wyckoff'] = wyckoff
                    data['CN_without_neigh_same_as_center'] = outer_CN

                    if metrics is not None:
                        combined = {**data, **metrics}
                        results.append(combined)
                    else:
                        results.append(data)

        return results
    except:
        print(traceback.format_exc())
    return 

if __name__ == "__main__":

    base_dir = os.getcwd()
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(d)]
    cif_dirs = [d for d in subdirs if any(f.endswith('.cif') for f in os.listdir(os.path.join(base_dir, d)))]

    if not cif_dirs:
        print("No directories with CIF files found.")
        exit(1)

    print("Select a directory to process:")
    for i, d in enumerate(cif_dirs, start=1):
        print(f"{i}. {d}")

    choice = input("Enter the number of the directory: ")
    try:
        selected_dir = cif_dirs[int(choice)-1]
    except (IndexError, ValueError):
        print("Invalid selection.")
        exit(1)

    root = os.path.join(base_dir, selected_dir)
    t0 = time.time()
    tasks = []
    results = []

    for file in os.listdir(root):
        if file.endswith(".cif"):
            path = os.path.join(root, file)
            result = find_cenvs_s(path)
            if result:
                results.extend(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{selected_dir}_RE_polyhedra_metrics.csv", index=False)
        print(f"Saved results to {selected_dir}_RE_polyhedra_metrics.csv")

    print(f"Total time: {round(time.time()-t0, 1)} seconds")