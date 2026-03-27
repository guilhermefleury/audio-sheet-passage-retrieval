import numpy as np
from pathlib import Path
import cv2

def load_npy_sheet(file_path, page=1):
    score_name = Path(file_path).name
    npy_path = Path(file_path) / "scores" / f"{score_name}_ly" / "coords" / f"systems_0{page}.npy"
    return np.load(npy_path)

def load_img_sheet(file_path, page=1):
    score_name = Path(file_path).name
    img_path = Path(file_path) / "scores" / f"{score_name}_ly" / "img" / f"0{page}.png"
    return cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

def img_crop(img,system_coords):
    # coordinates start from top-left corner and go clockwise
    # for each individual point we have (height = h, width = w)
    h_min = int(np.floor(system_coords[0][0]))
    h_max = int(np.floor(system_coords[2][0]))
    w_max = int(np.floor(system_coords[1][1]))
    w_min = int(np.floor(system_coords[3][1]))
    
    # Since the image slices must be 160x180, we need to adjust the cropping height
    if (h_max - h_min) != 160:
        h_center = (h_min + h_max) // 2
        h_min = h_center - 80
        h_max = h_center + 80
        
       
    return img[h_min:h_max, w_min:w_max]

def system_sheet_slicer(system, stride=90):    
    system_sequence = []
    w_iterator = 0
    while w_iterator + 180 <= system.shape[1]:
        snippet = system[:, w_iterator:w_iterator + 180]
        system_sequence.append(snippet)
        w_iterator += stride
    if w_iterator < system.shape[1]:
        snippet = system[:, system.shape[1]-180:system.shape[1]]
        system_sequence.append(snippet)
    return system_sequence

def get_sheet_systems(file_path, page=1,stride=90):
    img = load_img_sheet(file_path, page)
    system_coordinates = load_npy_sheet(file_path, page)
    n_systems = system_coordinates.shape[0]
    systems = []
    systems_sliced = []
    for system in range(n_systems):
        crop = img_crop(img, system_coordinates[system])
        systems.append(crop)
        slices = system_sheet_slicer(crop,stride)
        systems_sliced.append(slices)
    
    
    return systems, systems_sliced

def get_vertical_lines(img):
    """
    Extracts vertical lines from sheet music image using morphological filtering.
    Returns the filtered image showing only vertical structures.
    """
    # Threshold to binary (inverted: notes/lines become white)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Create vertical kernel and apply morphological opening
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    return vertical_lines

def count_bars_in_system(img, debug=False, min_distance=10):
    """
    Counts the number of bar lines in a system by detecting tall vertical structures.
    Merges nearby vertical lines (e.g., double bar lines) into single bars.
    
    Args:
        img: Grayscale image of a music system
        debug: If True, prints detailed contour information
        min_distance: Minimum horizontal distance (pixels) between separate bars (default: 10)
    """
    vertical_lines = get_vertical_lines(img)
    
    # Find contours in the filtered image
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"    Total contours found: {len(contours)}")
    
    # Extract x-positions of tall vertical lines (bar lines)
    bar_positions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if debug:
            print(f"    Contour: x={x}, y={y}, w={w}, h={h}")
        if h > 80:  # Bar lines are taller than 80 pixels
            bar_positions.append(x)
    
    # Sort positions left to right
    bar_positions.sort()
    
    # Merge nearby bars (e.g., double bar lines)
    merged_bars = []
    for x in bar_positions:
        if not merged_bars or (x - merged_bars[-1]) > min_distance:
            merged_bars.append(x)
        elif debug:
            print(f"    Merged bar at x={x} with previous bar at x={merged_bars[-1]}")
    
    if debug:
        print(f"    After merging: {len(merged_bars)} distinct bars")
    
    return len(merged_bars) - 1  # Return number of spaces between bars, not bar count

def get_bars_in_song(file_path):
    """
    Returns a list of bar counts for each system across all pages of a song.
    
    Args:
        file_path: Path to the MSMD piece directory (e.g., "../data/msmd/BachJS__BWV779__bach-invention-08")
    
    Returns:
        List of integers, where each element is the number of bars in that system (in order across all pages)
    """
    score_name = Path(file_path).name
    coords_dir = Path(file_path) / "scores" / f"{score_name}_ly" / "coords"
    
    # Find all systems_*.npy files to determine number of pages
    systems_files = sorted(coords_dir.glob("systems_*.npy"))
    
    if not systems_files:
        raise FileNotFoundError(f"No systems files found in {coords_dir}")
    
    bars_per_system = []
    
    # Iterate through each page
    for page_num in range(1, len(systems_files) + 1):
        # Get systems for this page
        systems, _ = get_sheet_systems(file_path, page_num)
        
        # Count bars in each system on this page
        for system_img in systems:
            bar_count = count_bars_in_system(system_img)
            bars_per_system.append(bar_count)
    
    return bars_per_system
    
    
if __name__ == "__main__":
    numpy = load_npy_sheet("../data/msmd/BachJS__BWV779__bach-invention-08", 1)
    img = load_img_sheet("../data/msmd/BachJS__BWV779__bach-invention-08", 1)
    systems = get_sheet_systems("../data/msmd/BachJS__BWV779__bach-invention-08", 1, stride=90)
    count = count_bars_in_system(img)
    print(f"Number of bar lines: {count}")