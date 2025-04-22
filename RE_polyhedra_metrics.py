import os
from io import StringIO
import contextlib
import time
from tqdm import tqdm
from cifkit import Cif
import traceback
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from skspatial.objects import Points
from collections import defaultdict
from numpy.linalg import norm
from re import match
from collections import deque
from multiprocessing import Pool, cpu_count

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

def compute_largest_face_area(coords, hull, angle_threshold_deg=0.05):
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
    face_centers = []
    for face_group in final_groups:
        face_area = sum(triangle_area(*coords[hull.simplices[idx]]) for idx in face_group)
        largest_area = max(largest_area, face_area)
        all_vertices = np.vstack([coords[hull.simplices[idx]] for idx in face_group])
        face_center = np.mean(all_vertices, axis=0)
        face_centers.append(face_center)

    from collections import Counter
    face_counts = Counter(len(group) for group in final_groups)
    face_descriptions = [f"{count}*{polygon_name(num_triangles)}" for num_triangles, count in sorted(face_counts.items(), reverse=True)]
    face_description_str = ', '.join(face_descriptions)
    
    return round(largest_area, 3), len(final_groups), face_description_str, face_centers

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

def d_by_dmin_CN_wo_center_type(k, v, center, site_symbol_map, cif_path):
    
    v = sorted(v, key=lambda x: x[1])
    i_same_as_center = min([100, *[i for i in range(len(v)) if site_symbol_map[v[i][0]] == site_symbol_map[k]]])
    points_wd = [[p[3], p[1], p[0]] for p in v if site_symbol_map[p[0]] not in RE_ELEMENTS]

    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:20]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    inner_gap_index = np.array(ind_gaps[::-1])
    inner_gap_index = inner_gap_index[inner_gap_index <= i_same_as_center]
    outer_CN_wo_center_type = int(inner_gap_index[0]) + 1 if len(inner_gap_index) else 0

    error_capture = StringIO()
    with contextlib.redirect_stdout(error_capture), contextlib.redirect_stderr(error_capture):
            try:
                distances, angles = polyhedron_metrics(center, points_wd[:outer_CN_wo_center_type])
                distances = [p[1] for p in points_wd[:outer_CN_wo_center_type]]
                coords = [p[0] for p in points_wd[:outer_CN_wo_center_type]]
                coords.append(center)
                coords = np.array(coords)
                central_atom_coord = coords[-1]
                neighbor_coords = coords[:-1]
                hull = ConvexHull(neighbor_coords)

                volume = hull.volume
                avg_point = np.mean(neighbor_coords, axis=0)
                distance_from_avg_point_to_center = np.linalg.norm(avg_point - central_atom_coord)

                largest_face_area, num_faces, face_geometry, face_centers = compute_largest_face_area(coords, hull)
                filtered_face_centers = [fc for fc in face_centers if np.linalg.norm(fc - central_atom_coord) > 1e-4]
                if filtered_face_centers:
                    distances_to_faces = np.linalg.norm(np.array(filtered_face_centers) - central_atom_coord, axis=1)
                    shortest_distance_to_face = np.min(distances_to_faces)
                else:
                    shortest_distance_to_face = None

                return {'outer_CN_wo_center_type': outer_CN_wo_center_type,
                        'formula_outer_CN_wo_center_type': get_formula(points_wd, outer_CN_wo_center_type, site_symbol_map),
                        # 'inner_CN': inner_CN,
                        # 'formula_inner_CN': get_formula(points_wd, inner_CN, site_symbol_map),
                        # 'lowest_inner_CN': lowest_inner_CN,
                        # 'formula_lowest_inner_CN': get_formula(points_wd, lowest_inner_CN, site_symbol_map),
                        # 'distances': distances,
                        # 'angles': angles,
                        'volume_of_polyhedron_wo': round(volume, 4),
                        'distance_from_avg_point_to_center_wo': round(distance_from_avg_point_to_center, 4),
                        'shortest_distance_to_face_wo': round(shortest_distance_to_face, 4) if shortest_distance_to_face is not None else None,
                        'largest_face_area_wo': largest_face_area,
                        'num_reconstructed_faces_wo': num_faces,
                        'face_geometry_wo': face_geometry
                        }
            except Exception as e:
                return {'file': os.path.basename(cif_path), 'site': k, 'error': f"Polyhedron construction failed: {str(e)}"}

    return 
        
def d_by_dmin(k, v, center, site_symbol_map, cif_path):
    
    points_wd =[[p[3], p[1], p[0]] for p in v]
    inner_CN = None
    
    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:20]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    
    inner_gap_index = np.array(ind_gaps[::-1])
    outer_CN = int(inner_gap_index[0]) + 1
    inner_gap_index = inner_gap_index - inner_gap_index[0]
    inner_gap_index = inner_gap_index[:5]
    inner_gap_index = inner_gap_index[inner_gap_index < 0]
    inner_gap_index += outer_CN
    inner_gap_index = inner_gap_index[inner_gap_index >= 4]

    if match(r"[A-Za-z]+", k).group() in RE_ELEMENTS:
        if not len(inner_gap_index):
            inner_CN = outer_CN
            lowest_inner_CN = outer_CN
        else:
            inner_CN = int(inner_gap_index[0])
            lowest_inner_CN = int(inner_gap_index.min())

        error_capture = StringIO()
        with contextlib.redirect_stdout(error_capture), contextlib.redirect_stderr(error_capture):
            try:
                distances, angles = polyhedron_metrics(center, points_wd[:outer_CN])
                distances = [p[1] for p in points_wd[:outer_CN]]
                coords = [p[0] for p in points_wd[:outer_CN]]
                coords.append(center)
                coords = np.array(coords)
                central_atom_coord = coords[-1]
                neighbor_coords = coords[:-1]
                hull = ConvexHull(neighbor_coords)

                volume = hull.volume
                avg_point = np.mean(neighbor_coords, axis=0)
                distance_from_avg_point_to_center = np.linalg.norm(avg_point - central_atom_coord)

                data_wo = d_by_dmin_CN_wo_center_type(k, v, center, site_symbol_map, cif_path)

                largest_face_area, num_faces, face_geometry, face_centers = compute_largest_face_area(coords, hull)
                filtered_face_centers = [fc for fc in face_centers if np.linalg.norm(fc - central_atom_coord) > 1e-4]
                if filtered_face_centers:
                    distances_to_faces = np.linalg.norm(np.array(filtered_face_centers) - central_atom_coord, axis=1)
                    shortest_distance_to_face = np.min(distances_to_faces)
                else:
                    shortest_distance_to_face = None

                return {'site': k,
                        'outer_CN': outer_CN,
                        'formula_outer_CN': get_formula(points_wd, outer_CN, site_symbol_map),
                        # 'inner_CN': inner_CN,
                        # 'formula_inner_CN': get_formula(points_wd, inner_CN, site_symbol_map),
                        # 'lowest_inner_CN': lowest_inner_CN,
                        # 'formula_lowest_inner_CN': get_formula(points_wd, lowest_inner_CN, site_symbol_map),
                        # 'distances': distances,
                        # 'angles': angles,
                        'volume_of_polyhedron': round(volume, 4),
                        'distance_from_avg_point_to_center': round(distance_from_avg_point_to_center, 4),
                        'shortest_distance_to_face': round(shortest_distance_to_face, 4) if shortest_distance_to_face is not None else None,
                        'largest_face_area': largest_face_area,
                        'num_reconstructed_faces': num_faces,
                        'face_geometry': face_geometry
                        }, data_wo
            except Exception as e:
                return {'file': os.path.basename(cif_path), 'site': k, 'error': f"Polyhedron construction failed: {str(e)}"}
    
    return

def find_cenvs_s(cif_path):
    try:
        results = []
        cif = Cif(cif_path, is_formatted=True)
        # label, symbol, mult, Wyckoff_symbol, x, y, z, occ
        error_capture = StringIO()
        with contextlib.redirect_stdout(error_capture):
            cif.compute_connections()
        connection_output = error_capture.getvalue()
        poly_error_lines = [line.strip() for line in connection_output.splitlines() if "Error in polyhedron calculation" in line]
        import re
        for msg in poly_error_lines:
            match_site = re.search(r"for(\w+)\s", msg)
            site = match_site.group(1) if match_site else None
            results.append({'file': os.path.basename(cif_path), 'site': site, 'error': msg})
        conns = cif.connections
        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

        data = None
        for i, (k, v) in enumerate(conns.items()):
            if match(r"[A-Za-z]+", k).group() in RE_ELEMENTS:
                wyckoff = f"{loop_vals[2][i]}{loop_vals[3][i]}"
                central_atom = loop_vals[1][i]
                center = v[0][2]
                data, data_wo = d_by_dmin(k, v, center, site_symbol_map, cif_path)
            
                if data is not None:
                    data['CIF#'] = cif_path.split(os.sep)[-1][:-4]
                    data['central_atom'] = central_atom
                    data['Wyckoff'] = wyckoff
                    # data['CN_without_neigh_same_as_center'] = outer_CN_wo_center_type

                    if data_wo is not None:
                        data_wo = {f"{k}": v for k, v in data_wo.items() if k not in {'site', 'error', 'file'}}
                        combined = {**data, **data_wo}
                        results.append(combined)
                    else:
                        results.append(data)

                else:
                    results.append({'file': os.path.basename(cif_path), 'site': k, 'error': 'Polyhedron metrics returned None'})

        return results
    except Exception as e:
        return [{'file': os.path.basename(cif_path), 'site': None, 'error': str(e)}]

def worker(file_path):
    try:
        return find_cenvs_s(file_path)
    except Exception as e:
        return [{'file': os.path.basename(file_path), 'site': None, 'error': str(e)}]

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(d)]
    cif_dirs = [d for d in subdirs if any(f.endswith('.cif') for f in os.listdir(os.path.join(base_dir, d)))]

    if not cif_dirs:
        user_choice = input("No directories with CIF files found. Specify a directory manually? (Y/n): ").strip().lower()
        if user_choice == 'y' or user_choice == '':
            selected_dir = input("Enter the directory path: ").strip()
            if not os.path.isdir(selected_dir):
                print("Directory not found.")
                exit(1)
        else:
            exit(1)

    print("Select a directory to process:")
    for i, d in enumerate(cif_dirs, start=1):
        print(f"{i}. {d}")

    choice = input("Enter the number of the directory or manually specify a directory: ")
    try:
        selected_dir = cif_dirs[int(choice)-1]
    except (IndexError, ValueError):
        if os.path.isdir(choice.strip()):
            selected_dir = choice.strip()
        else:
            print("Invalid selection or directory not found.")
            exit(1)

    root = os.path.join(base_dir, selected_dir)
    t0 = time.time()
    tasks = []
    results = []
    errors = []

    with Pool(processes=cpu_count() - 1) as pool:
        cif_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".cif")]
        all_results = []
        for result in tqdm(pool.imap_unordered(worker, cif_files), total=len(cif_files)):
            if result:
                all_results.append(result)

    for result in all_results:
        if result:
            for entry in result:
                if 'error' in entry:
                    errors.append({k: entry.get(k) for k in ('file', 'site', 'error')})
                else:
                    results.append(entry)

    if results:
        df = pd.DataFrame(results)
        output_filename = f"{os.path.basename(os.path.normpath(selected_dir))}_RE_polyhedra_metrics.csv"
        output_path = os.path.join(base_dir, output_filename)
        df.to_csv(output_path, index=False)

    if errors:
        df_err = pd.DataFrame(errors)
        errors_output_filename = f"{os.path.basename(os.path.normpath(selected_dir))}_RE_polyhedra_errors.csv"
        errors_output_path = os.path.join(base_dir, errors_output_filename)
        df_err.to_csv(errors_output_path, index=False)

    # Create summary with min/max values per CIF# from data_wo
    if results:
        grouped_by_cif = defaultdict(list)
        for row in results:
            if all(key in row for key in ['volume_of_polyhedron_wo', 'distance_from_avg_point_to_center_wo', 'shortest_distance_to_face_wo', 'largest_face_area_wo']):
                grouped_by_cif[row['CIF#']].append(row)

        summary_rows = []
        for cif_name, entries in grouped_by_cif.items():
            def safe_min(key): return min(e[key] for e in entries if e[key] is not None)
            def safe_max(key): return max(e[key] for e in entries if e[key] is not None)

            summary_rows.append({
                'CIF#': cif_name,
                'max_volume_of_polyhedron': safe_max('volume_of_polyhedron_wo'),
                'min_volume_of_polyhedron': safe_min('volume_of_polyhedron_wo'),
                'max_distance_from_avg_point_to_center': safe_max('distance_from_avg_point_to_center_wo'),
                'min_distance_from_avg_point_to_center': safe_min('distance_from_avg_point_to_center_wo'),
                'max_shortest_distance_to_face': safe_max('shortest_distance_to_face_wo'),
                'min_shortest_distance_to_face': safe_min('shortest_distance_to_face_wo'),
                'max_largest_face_area': safe_max('largest_face_area_wo'),
                'min_largest_face_area': safe_min('largest_face_area_wo')
            })

        df_summary = pd.DataFrame(summary_rows)
        summary_output_filename = f"{os.path.basename(os.path.normpath(selected_dir))}_RE_polyhedra_summary.csv"
        summary_output_path = os.path.join(base_dir, summary_output_filename)
        df_summary.to_csv(summary_output_path, index=False)

    print(f"Total time: {round(time.time()-t0, 1)} seconds")