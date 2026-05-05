import cv2
from sheet import get_sheet_systems, get_vertical_lines, count_bars_in_system

# Load all systems from the page
piece_path = "../data/msmd/BachJS__BWV779__bach-invention-08"
page = 1

systems, systems_sliced = get_sheet_systems(piece_path, page)
print(f"Total systems on page {page}: {len(systems)}")
print("=" * 60)

# Test each system
for system_index in range(len(systems)):
    system_img = systems[system_index]
    
    print(f"\nSystem #{system_index}:")
    print(f"  Shape: {system_img.shape}")
    
    # Apply vertical line filter
    vertical_lines = get_vertical_lines(system_img)
    
    # Count bars in this system
    bar_count = count_bars_in_system(system_img, debug=True)
    print(f"  Bar lines detected: {bar_count}")
    
    # Visualize each system
    cv2.imshow(f"System {system_index} - Original", system_img)
    cv2.imshow(f"System {system_index} - Vertical Lines", vertical_lines)
    
    # Create result with count
    system_color = cv2.cvtColor(system_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(system_color, f"System {system_index}: {bar_count} bars", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow(f"System {system_index} - Result", system_color)
    
    # Save results
    cv2.imwrite(f"system_{system_index}_original.png", system_img)
    cv2.imwrite(f"system_{system_index}_vertical.png", vertical_lines)
    cv2.imwrite(f"system_{system_index}_result.png", system_color)

print("\n" + "=" * 60)
print(f"Total bar lines across all systems: {sum(count_bars_in_system(s) for s in systems)}")
print("\nPress any key to close all windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nAll results saved!")


