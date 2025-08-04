# (c) Antoni Pech Alberich 2024
import os
import glob
import urllib.request as ur
import tarfile
import numpy as np
import math
from frame.geometry.strop import Polygon
from frame.geometry.geometry import Point


PointSequence = list[Point] | list[np.ndarray]
Rectangle = list[float]
GBFACTOR = float(1 << 30)


def is_dataset_downloaded(path: str, folder_name: str) -> bool:
    """
    Checks if the dataset is considered downloaded by verifying the existence
    of at least 100 'config*' or 'worker*' files in the specified folder.

    Args:
        path (str): The base path to the dataset.
        folder_name (str): The name of the folder containing the dataset.

    Returns:
        bool: True if the dataset is downloaded, False otherwise.
    """
    config_files = glob.glob(os.path.join(path, folder_name, "config*"))
    worker_files = glob.glob(os.path.join(path, folder_name, "worker*"))

    has_enough_config_files = len(config_files) >= 100
    has_enough_worker_files = len(worker_files) >= 100

    return has_enough_config_files or has_enough_worker_files


def decide_download(url: str) -> bool:
    """
    Decides whether to download a file based on its size.

    Args:
        url (str): The URL of the file to check.

    Returns:
        bool: True if the user agrees to proceed with the download, False otherwise.
    """
    try:
        # Open the URL and retrieve the content length
        response = ur.urlopen(url)
        size_gb = int(response.info()["Content-Length"]) / GBFACTOR
    except KeyError:
        print("Error: Unable to determine file size.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    # Confirm with the user if the file is larger than 1GB
    if size_gb > 1:
        user_input = (
            input(
                f"This file is {size_gb:.2f} GB. Do you want to proceed with the download? (y/N): "
            )
            .strip()
            .lower()
        )
        return user_input == "y"
    else:
        return True


def download_dataset(root: str, url: str, file_name: str | None = None):
    """
    Downloads and extracts a dataset from the specified URL.

    Args:
        root (str): The root directory where the dataset will be saved.
        url (str): The URL of the dataset to download.
        file_name (str, optional): The desired name of the downloaded file.
                                   If None, it is inferred from the URL.

    Returns:
        None
    """
    # Infer file name if not provided
    if file_name is None:
        file_name = os.path.basename(url)

    # Full path to the file
    f_name = os.path.join(root, file_name)

    # Check if the file already exists or confirm download
    if not os.path.exists(f_name) and decide_download(url):
        data = ur.urlopen(url)
        size = int(data.info()["Content-Length"])
        chunk_size = 1024 * 1024  # 1 MB per chunk
        num_iter = int(size / chunk_size) + 2
        downloaded_size = 0

        with open(f_name, "wb") as f:
            for _ in range(num_iter):
                chunk = data.read(chunk_size)
                if not chunk:  # Stop if no more data
                    break
                downloaded_size += len(chunk)
                f.write(chunk)
    else:
        print("Tar file already downloaded or download canceled.")

    # Extract the file
    print("Downloaded floorplan data to", f_name)
    print("Unpacking. This may take a while.")
    with tarfile.open(f_name) as file:
        file.extractall(root)

    # Clean up
    os.remove(f_name)
    print("Dataset unpacked successfully.")


def weight_sum(
    connections_b2b: np.ndarray, connections_p2b: np.ndarray, target_id: int
) -> float:
    """
    Calculate the weighted sum of connections associated with a target block.

    Args:
        connections_b2b (np.ndarray): A 2D array representing block-to-block connections.
            Each row contains [block_id_1, block_id_2, weight].
        connections_p2b (np.ndarray): A 2D array representing pin-to-block connections.
            Each row contains [pin_id, block_id, weight].
        target_id (int): The ID of the target block to calculate the weighted sum for.

    Returns:
        float: The total weight of block-to-block and block-related pin connections
    """
    # Identify rows in block-to-block connections that involve the target block
    mask = (connections_b2b[:, 0] == target_id) | (connections_b2b[:, 1] == target_id)
    result = np.sum(connections_b2b[mask, 2])

    # Identify rows in pin-to-block connections where the target is a pin or block
    mask_block = connections_p2b[:, 1] == target_id
    result += np.sum(connections_p2b[mask_block, 2])

    return float(result)


def rescale(value: float, old_min: float = 0, old_max=1, new_min=500, new_max=1500):
    assert value >= 0, "Formula do not hold for negative values"
    assert abs(old_max - old_min) > 1e-6, "Division by 0!"
    new_value = 500 + (value - old_min) * (new_max - new_min) / (old_max - old_min)
    return new_value


def compute_perimeter(vertices: PointSequence) -> float:
    """
    Compute the perimeter of a polygon given its vertices.

    Args:
        vertices (PointSequence): A list or sequence of 2D points (tuples) representing the polygon's vertices.
            The vertices are assumed to be ordered consecutively.

    Returns:
        float: The perimeter of the polygon.
    """
    perimeter = 0.0

    # Iterate over consecutive vertices to calculate edge lengths
    for i in range(len(vertices) - 1):
        p1 = vertices[i]
        p2 = vertices[i + 1]
        if isinstance(p1, Point):
            distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        else:
            distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        perimeter += distance

    return float(perimeter)


def strop_decomposition(vertices: PointSequence) -> list[Rectangle]:
    """
    Decomposes a polygon into Single-Trunk-Rectilinear-Orthogonal-Polygons.

    Args:
        vertices: List of dataclass Point or numpy.ndarray [p1, p2, ...]
            representing the polygon's vertices.

    Returns:
        rectangles: List of rectangles represented as [x,y,w,h], the center
            (x,y), the width w, and the height h.
    """
    rectangles = list[Rectangle]()
    # Extract unique x and y coordinates and sort them
    x_coords = sorted(set(p.x if isinstance(p, Point) else p[0] for p in vertices))
    y_coords = sorted(
        set(p.y if isinstance(p, Point) else p[1] for p in vertices), reverse=True
    )

    rows = len(y_coords) - 1
    cols = len(x_coords) - 1

    # Check each rectangle defined by consecutive x and y intervals
    m = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            # Define the rectangle bounds
            x_min, x_max = x_coords[j], x_coords[j + 1]
            y_max, y_min = y_coords[i], y_coords[i + 1]
            # Determine the center of the rectangle
            center_x = float((x_min + x_max) / 2)
            center_y = float((y_min + y_max) / 2)
            # Check if the center is inside the polygon
            if is_point_inside_polygon(Point(center_x, center_y), vertices):
                m[i,j] = 1

    s = Polygon(m)
    assert len(s.instances) > 0, f"Polygon has no STROPs {vertices}"
    sol = s.instances[0]
    # Extract rectangles from the STROP instance
    for r in sol.rectangles():
        x_min, x_max = x_coords[r.columns.low], x_coords[r.columns.high + 1]
        y_max, y_min = y_coords[r._rows.low], y_coords[r._rows.high + 1]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        rectangles.append([float(cx), float(cy), float(w), float(h)])

    return rectangles


def is_point_inside_polygon(point: Point, vertices: PointSequence) -> bool:
    """
    Determine if a point is inside a polygon using the even-odd rule algorithm.

    Args:
        point: the point tuple (x,y) to check.
        vertices: a list of points representing the vertices of the polygon.

    Returns:
        A boolean whether the point is inside or not.
    """

    n = len(vertices)
    inside = False

    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        # p1 p2 are Points or ndarrays
        if isinstance(p1, np.ndarray):
            p1 = Point(float(p1[0]), float(p1[1]))
        if isinstance(p2, np.ndarray):
            p2 = Point(float(p2[0]), float(p2[1]))

        if (p1.y <= point.y < p2.y) or (p2.y <= point.y < p1.y):
            intersect_x = p1.x + (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y)
            if point.x < intersect_x:
                inside = not inside

    return inside


def compute_centroid(partition: List[Rectangle]) -> Point:
    """
    Compute the centroid of a simple polygon.

    Args:
        vertices: List of tuples [(x1,y1,w1,h1), ..., (xn, yn, wn, hn)]
                representing the polygon's vertices. The polygon should be
                closed (first vertex = last vertex).

    Returns:
        A tuple (Cx, Cy) representing the centroid of the polygon.
    """
    assert len(partition) > 0, "The partition should be not empty."
    n = len(partition)
    cx = 0.0
    cy = 0.0
    total_area = 0.0
    for i in range(n):
        x, y, w, h = partition[i]
        cx += w * h * x
        cy += w * h * y
        total_area += w * h
    cx /= total_area
    cy /= total_area
    return Point(float(cx), float(cy))
