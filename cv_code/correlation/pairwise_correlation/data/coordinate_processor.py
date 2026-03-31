"""
Coordinate processing utilities.
"""

import ast


def process_coordinates(row):
    """
    Process coordinate data from a DataFrame row.
    
    Args:
        row: DataFrame row containing 'X' and 'Y' columns with coordinate data
        
    Returns:
        List of filtered coordinate pairs [(x, y), ...]
    """
    try:
        x_coords = ast.literal_eval(str(row['X']).strip())
        y_coords = ast.literal_eval(str(row['Y']).strip())

        # print(f"Parsed X: {x_coords}, Parsed Y: {y_coords}")
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse: {row['X']}, {row['Y']} — Error: {e}")
        return []

    filtered = []
    for x, y in zip(x_coords, y_coords):
        # print(f"Checking pair: ({x}, {y})")
        if not (x == 0 and y == 0):
            filtered.append([x, y])

    # print(f"Filtered result: {filtered}")
    return filtered
