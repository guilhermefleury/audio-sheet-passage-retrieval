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

def system_sheet_slicer(system, step=90):    
    system_sequence = []
    w_iterator = 0
    while w_iterator + 180 <= system.shape[1]:
        snippet = system[:, w_iterator:w_iterator + 180]
        system_sequence.append(snippet)
        w_iterator += step
    if w_iterator < system.shape[1]:
        snippet = system[:, system.shape[1]-180:system.shape[1]]
        system_sequence.append(snippet)
    return system_sequence

def get_sheet_systems(file_path, page=1):
    img = load_img_sheet(file_path, page)
    system_coordinates = load_npy_sheet(file_path, page)
    n_systems = system_coordinates.shape[0]
    systems = []
    for system in range(n_systems):
        crop = img_crop(img, system_coordinates[system])
        slices = system_sheet_slicer(crop)
        systems.append(slices)
    
    
    return systems



#-----------------------------------------------------------------------------

numpy = load_npy_sheet("../data/msmd/BachJS__BWV779__bach-invention-08", 1)
img = load_img_sheet("../data/msmd/BachJS__BWV779__bach-invention-08", 1)
crop = img_crop(img, numpy[0])
system =system_sheet_slicer(crop)
systems = get_sheet_systems("../data/msmd/BachJS__BWV779__bach-invention-08")
print(f"Number of systems in the page: {len(systems)}")
print(f"Number of slices in the first system: {len(systems[0])}")
print(f"Shape of each slice: {systems[0][0].shape}")
print(f"Shape of the original image: {img.shape}")
print(f"Shape of the first system crop: {crop.shape}")
print(f"Shape of the first slice of the first system: {system[0].shape}")
print(f"First slice array data:\n {system[0]}")
print(f"Systems variable dimension: {len(systems)} systems, each with {len(systems[0])} slices.")


cols = 6
pad_img = np.full_like(system[0], 255)  # white pad
chunks = [system[i:i+cols] for i in range(0, len(system), cols)]
if len(chunks[-1]) < cols:
    chunks[-1] += [pad_img] * (cols - len(chunks[-1]))
rows = [cv2.hconcat(ch) for ch in chunks]
montage = cv2.vconcat(rows)
cv2.imshow("All slices (grid)", montage)
cv2.waitKey(0); cv2.destroyAllWindows()