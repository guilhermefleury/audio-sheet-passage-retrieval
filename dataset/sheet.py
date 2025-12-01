import numpy as np
from pathlib import Path
import cv2

def load_npy_sheet(file_path, page=1):
    score_name = file_path.split("/")[-1]
    npy_path = Path(file_path) / "scores" / f"{score_name}_ly" / "coords" / f"systems_0{page}.npy"
    return np.load(npy_path)

def load_img_sheet(file_path, page=1):
    score_name = file_path.split("/")[-1]
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

def count_bars_in_system(img):
    vertical_lines = get_vertical_lines(img)
    return len(vertical_lines)-1

#-----------------------------------------------------------------------------

numpy = load_npy_sheet("../data/msmd/BachJS__BWV779__bach-invention-08", 1)
img = load_img_sheet("../data/msmd/BachJS__BWV779__bach-invention-08", 1)
systems = get_sheet_systems("../data/msmd/BachJS__BWV779__bach-invention-08", 1, stride=90)
count=count_bars_in_system(img)
print(f"Number of bar lines: {count}")