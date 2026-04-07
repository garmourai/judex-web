"""
Shuttle matching algorithms and cost matrix computation.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _extract_vec(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size < 3:
        raise ValueError(f"Expected at least 3 elements in vector, got {arr.size}")
    return arr[:3].reshape(3, 1)


def _get_R_t(cam):
    if cam.rotation_matrix is not None and cam.translation_vectors is not None:
        R = np.asarray(cam.rotation_matrix, dtype=float)
        t = np.asarray(cam.translation_vectors, dtype=float).reshape(3, 1)
    else:
        rvec = _extract_vec(cam.calibration_rotation_vectors)
        R, _ = cv2.Rodrigues(rvec)
        t = _extract_vec(cam.calibration_translation_vectors)
    return R, t


def greedy_min_cost_assignment_with_disambiguation(
    pts1,
    pts2,
    cost_matrix_formula1,
    cost_matrix_formula2,
    ambiguity_thresh=3.0,
    frame_num=None,
    return_diagnostics=False,
):
    num_rows, num_cols = cost_matrix_formula1.shape
    assigned_src = set()
    assigned_sink = set()
    assignments = []
    diagnostics = []

    all_pairs = [(i, j, cost_matrix_formula1[i, j]) for i in range(num_rows) for j in range(num_cols)]
    all_pairs.sort(key=lambda x: x[2])

    # print(f"=================> For frame num: ", frame_num)

    # print("pts1: ", pts1)
    # print("pts2: ", pts2)

    for i, j, cost in all_pairs:
        if i in assigned_src or j in assigned_sink:
            continue

        # Check for ambiguity: are there other options with similar cost sharing same src or sink?
        ambiguous = False
        for j_alt in range(num_cols):
            if j_alt != j and abs(cost - cost_matrix_formula1[i, j_alt]) < ambiguity_thresh and j_alt not in assigned_sink:
                ambiguous = True
                break
        for i_alt in range(num_rows):
            if i_alt != i and abs(cost - cost_matrix_formula1[i_alt, j]) < ambiguity_thresh and i_alt not in assigned_src:
                ambiguous = True
                break

        if ambiguous:
            # Look for the best alternative using formula 2 cost
            best_i, best_j = i, j
            best_cost = cost_matrix_formula2[i, j]
            # print(f"formula 2 used for {i} and {j}, cost: {best_cost}")

            for j_alt in range(num_cols):
                if j_alt not in assigned_sink:
                    alt_cost = cost_matrix_formula1[i, j_alt]
                    # Removed ambiguity_thresh condition - always consider alternatives
                    if cost_matrix_formula2[i, j_alt] < best_cost:
                        best_j = j_alt
                        best_i = i
                        best_cost = cost_matrix_formula2[i, j_alt]
                        # print(f"formula 2 used for {i} and {j_alt}, cost: {best_cost}")

            for i_alt in range(num_rows):
                if i_alt not in assigned_src:
                    alt_cost = cost_matrix_formula1[i_alt, j]
                    # Removed ambiguity_thresh condition - always consider alternatives
                    if cost_matrix_formula2[i_alt, j] < best_cost:
                        best_i = i_alt
                        best_j = j
                        best_cost = cost_matrix_formula2[i_alt, j]
                        # print(f"formula 2 used for {i_alt} and {j}, cost: {best_cost}")

            # print(f"using formula 2 got this matching: {best_i} and {best_j}")
            if best_i not in assigned_src and best_j not in assigned_sink:
                if return_diagnostics:
                    diagnostics.append({
                        "selected_i": int(best_i),
                        "selected_j": int(best_j),
                        "initial_i": int(i),
                        "initial_j": int(j),
                        "is_ambiguous": True,
                    })
                assignments.append((best_i, best_j))
                assigned_src.add(best_i)
                assigned_sink.add(best_j)
        else:
            if return_diagnostics:
                diagnostics.append({
                    "selected_i": int(i),
                    "selected_j": int(j),
                    "initial_i": int(i),
                    "initial_j": int(j),
                    "is_ambiguous": False,
                })
            assignments.append((i, j))
            assigned_src.add(i)
            assigned_sink.add(j)

        if len(assignments) >= min(num_rows, num_cols):
            break

    # if frame_num >= 2512:
    #     breakpoint()

    if return_diagnostics:
        return assignments, diagnostics
    return assignments


def compute_fundamental_matrix(cam1, cam2):
    R1, t1 = _get_R_t(cam1)
    R2, t2 = _get_R_t(cam2)

    R_rel = R2 @ R1.T
    t_rel = t2 - (R_rel @ t1)

    t_x = np.array([
        [0, -t_rel[2, 0], t_rel[1, 0]],
        [t_rel[2, 0], 0, -t_rel[0, 0]],
        [-t_rel[1, 0], t_rel[0, 0], 0],
    ])

    E = t_x @ R_rel

    K1_inv = np.linalg.inv(np.asarray(cam1.camera_matrix, float))
    K2_inv_T = np.linalg.inv(np.asarray(cam2.camera_matrix, float)).T
    F = K2_inv_T @ E @ K1_inv

    return F / F[2, 2]


def build_cost_matrix(pts1, pts2, F, cam1, cam2, alpha=1.0, beta=1.0, gamma=1.0,
                     prev_pts1=None, prev_pts2=None, prev_matches=None,
                     prev_frame_gap: int = 1, max_frame_gap_for_temporal: int = 5):
    """
    pts1 and pts2 are undistorted pixel coordinates.
    prev_pts1 and prev_pts2 are previous frame coordinates for temporal consistency.
    Returns:
        - Cost matrix C
        - Reprojection error matrix rho_matrix
        - Epipolar distance matrix epipolar_matrix
        - Temporal cost matrix temporal_matrix
    """

    P1 = np.asarray(cam1.projection_matrix, float)
    P2 = np.asarray(cam2.projection_matrix, float)

    N, M = len(pts1), len(pts2)
    C = np.zeros((N, M), float)
    rho_matrix = np.zeros((N, M), float)
    epipolar_matrix = np.zeros((N, M), float)
    temporal_matrix = np.zeros((N, M), float)

    def reproj(P, X, orig):
        xh = P @ np.vstack((X, [1]))
        x = (xh[:2] / xh[2]).ravel()
        return np.linalg.norm(x - orig)

    def compute_temporal_cost(curr_pt1, curr_pt2):
        """Compute temporal consistency using previous matched pairs; ignore if frame gap too large."""
        if prev_pts1 is None or prev_pts2 is None or prev_matches is None or len(prev_matches) == 0:
            return 0.0
        if prev_frame_gap is None or prev_frame_gap > max_frame_gap_for_temporal:
            return 0.0
        if len(prev_pts1) == 0 or len(prev_pts2) == 0:
            return 0.0

        p1 = np.array(curr_pt1, dtype=float)
        p2 = np.array(curr_pt2, dtype=float)
        prev1 = np.array(prev_pts1, dtype=float)
        prev2 = np.array(prev_pts2, dtype=float)

        best = None
        for k, l in prev_matches:
            if k < 0 or l < 0:
                continue
            if k >= prev1.shape[0] or l >= prev2.shape[0]:
                continue
            d1 = float(np.linalg.norm(prev1[k] - p1))
            d2 = float(np.linalg.norm(prev2[l] - p2))
            s = d1 + d2
            if best is None or s < best:
                best = s

        if best is None:
            return 0.0

        gap_scale = 1.0 / max(1, prev_frame_gap)
        return best * gap_scale

    for i, (x1, y1) in enumerate(pts1):
        x1h = np.array([x1, y1, 1.0])
        l2 = F @ x1h
        denom = np.hypot(l2[0], l2[1]) + 1e-12  # avoid division by zero

        for j, (x2, y2) in enumerate(pts2):
            x2h = np.array([x2, y2, 1.0])
            ep_dist = abs(x2h @ l2) / denom
            epipolar_matrix[i, j] = ep_dist

            pt1 = np.array([[x1], [y1]], float)
            pt2 = np.array([[x2], [y2]], float)
            X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
            X = (X_h / X_h[3])[:3]

            rho = reproj(P1, X, (x1, y1)) + reproj(P2, X, (x2, y2))
            rho_matrix[i, j] = rho
            
            # Compute temporal cost (index-agnostic, uses nearest previous points)
            temporal_cost = compute_temporal_cost((x1, y1), (x2, y2))
            temporal_matrix[i, j] = temporal_cost
            
            formula_1 = rho
            formula_2 = beta * rho + gamma * temporal_cost
            
            C[i, j] = formula_1

    return C, rho_matrix, epipolar_matrix, temporal_matrix


def match_shuttles(
    cam1,
    cam2,
    pts1,
    pts2,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    cost_threshold=100.0,
    prev_pts1=None,
    prev_pts2=None,
    prev_matches=None,
    prev_frame_gap: int = 1,
    max_frame_gap_for_temporal: int = 5,
    frame_num=0,
    return_diagnostics=False,
):
    """
    pts1, pts2: List of undistorted pixel coordinates (x, y).
    prev_pts1, prev_pts2: Previous frame coordinates for temporal consistency.
    """
    if not pts1 or not pts2:
        empty = np.zeros((len(pts1), len(pts2)))
        if return_diagnostics:
            return [], empty, empty, empty, empty, empty, []
        return [], empty, empty, empty, empty

    F = compute_fundamental_matrix(cam1, cam2)
    C, rho_matrix, epipolar_matrix, temporal_matrix = build_cost_matrix(
        pts1, pts2, F, cam1, cam2, alpha, beta, gamma,
        prev_pts1, prev_pts2, prev_matches,
        prev_frame_gap=prev_frame_gap,
        max_frame_gap_for_temporal=max_frame_gap_for_temporal
    )

    row_idx, col_idx = linear_sum_assignment(C)
    if len(row_idx) == 0:
        return [], C, rho_matrix, epipolar_matrix, temporal_matrix

    # Use formula2 (with temporal cost) for assignment
    formula2_matrix = alpha * epipolar_matrix + beta * rho_matrix + gamma * temporal_matrix
    assignment_result = greedy_min_cost_assignment_with_disambiguation(
        pts1,
        pts2,
        cost_matrix_formula1=C,
        cost_matrix_formula2=formula2_matrix,
        ambiguity_thresh=5.0,
        frame_num=frame_num,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        assignment, diagnostics = assignment_result
    else:
        assignment = assignment_result
        diagnostics = []
    if cost_threshold is not None:
        assignment = [(i, j) for (i, j) in assignment if C[i, j] <= cost_threshold]

    if return_diagnostics:
        return assignment, C, rho_matrix, epipolar_matrix, temporal_matrix, formula2_matrix, diagnostics
    return assignment, C, rho_matrix, epipolar_matrix, temporal_matrix
