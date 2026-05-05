from audio import get_spec_systems

file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"
performance_name = "BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804"

print("Loading and slicing spectrograms with 50% overlap...")
# stride_frames=10 means 50% overlap (10 frames stride for 20 frame window)
systems, systems_sliced = get_spec_systems(file_path, performance_name, window_frames=20, stride_frames=10)

print(f"\nTotal systems: {len(systems)}")
print(f"\n{'System':<8} {'Full Shape':<15} {'Duration':<12} {'Slices':<8} {'Slice Shape':<15}")
print("-" * 70)

for i, (system, slices) in enumerate(zip(systems, systems_sliced)):
    duration = system.shape[1] / 20.0
    slice_shape = slices[0].shape if slices else "N/A"
    print(f"{i:<8} {str(system.shape):<15} {duration:.2f}s{'':<7} {len(slices):<8} {str(slice_shape):<15}")

# Show details for first system
print(f"\n\n=== System 0 Details (50% overlap) ===")
print(f"Full system shape: {systems[0].shape}")
print(f"Number of 1-second slices: {len(systems_sliced[0])}")
print(f"Window: 20 frames (1 sec), Stride: 10 frames (0.5 sec)")
print(f"\nAll slice time ranges:")
for j, slice_spec in enumerate(systems_sliced[0]):
    # Calculate actual frame positions
    if j < len(systems_sliced[0]) - 1:
        start_frame = j * 10
    else:
        # Last slice is anchored at the end
        start_frame = 120 - 20
    end_frame = start_frame + 20
    start_time = start_frame / 20.0
    end_time = end_frame / 20.0
    print(f"  Slice {j}: {slice_spec.shape} - frames [{start_frame:3d}-{end_frame:3d}] = {start_time:.1f}s to {end_time:.1f}s")

# Test on final system (system 10) which has 134 frames (6.7 seconds)
print(f"\n\n=== System 10 Details (final system - 6.7s) ===")
print(f"Full system shape: {systems[10].shape}")
print(f"Number of 1-second slices: {len(systems_sliced[10])}")
print(f"Duration: {systems[10].shape[1] / 20.0:.2f}s")
print(f"\nAll slice time ranges:")
for j, slice_spec in enumerate(systems_sliced[10]):
    # For debug, let's just show what we have
    print(f"  Slice {j}: {slice_spec.shape}")

# Show actual frame coverage
print(f"\nActual frame positions for system 10:")
for j in range(len(systems_sliced[10])):
    if j < len(systems_sliced[10]) - 1:
        # Regular slices
        start_frame = j * 10
    else:
        # Last slice anchored at end
        start_frame = 134 - 20
    end_frame = start_frame + 20
    start_time = start_frame / 20.0
    end_time = end_frame / 20.0
    print(f"  Slice {j}: frames [{start_frame:3d}-{end_frame:3d}] = {start_time:.2f}s to {end_time:.2f}s")
