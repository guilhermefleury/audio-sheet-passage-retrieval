from sheet import get_sheet_systems, count_bars_in_system, load_img_sheet
import cv2

# Load page 2
file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"
systems, _ = get_sheet_systems(file_path, page=2)

print(f"Total systems on page 2: {len(systems)}")

# Check the last system (index 3)
last_system = systems[-1]
print(f"\nLast system shape: {last_system.shape}")

# Count with debug
print("\nDetailed analysis of last system:")
bar_count = count_bars_in_system(last_system, debug=True)
print(f"Final bar count returned: {bar_count}")

# Save visualization
cv2.imwrite("last_system_page2.png", last_system)
print("\nSaved visualization to last_system_page2.png")
