from audio import slice_spectrogram_by_systems, load_performance_spec

file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"
performance_name = "BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804"

# Load and slice
print("Loading and slicing spectrogram...")
system_specs = slice_spectrogram_by_systems(file_path, performance_name)

# Load full spec for comparison
full_spec = load_performance_spec(file_path, performance_name)

print(f"\nFull spectrogram shape: {full_spec.shape}")
print(f"Duration: {full_spec.shape[1] / 20:.2f} seconds at 20 fps")
print(f"\nNumber of systems: {len(system_specs)}")

print(f"\n{'System':<8} {'Shape':<15} {'Frames':<8} {'Duration (s)':<12}")
print("-" * 50)
total_frames = 0
for i, spec in enumerate(system_specs):
    duration = spec.shape[1] / 20.0
    total_frames += spec.shape[1]
    print(f"{i:<8} {str(spec.shape):<15} {spec.shape[1]:<8} {duration:<12.2f}")

print(f"\nTotal frames from systems: {total_frames}")
print(f"Original frames: {full_spec.shape[1]}")
print(f"Frames accounted for: {total_frames / full_spec.shape[1] * 100:.1f}%")


