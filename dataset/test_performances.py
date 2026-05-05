from audio import get_performance_list, load_performance_spec, load_all_performance_specs

file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"

# Get list of all performances
performances = get_performance_list(file_path)
print(f"Total performances: {len(performances)}\n")

# Show first few performances
print("First 5 performances:")
for perf in performances[:5]:
    print(f"  - {perf}")

# Load spectrogram from one specific performance
print(f"\n\nLoading spectrogram from: {performances[0]}")
spec = load_performance_spec(file_path, performances[0])

print(f"\nSpectrogram shape: {spec.shape}")
print(f"Duration: {spec.shape[1] / 20:.2f} seconds (at 20 fps)")

# Example: Load all performance spectrograms
print("\n\n--- Loading all performance spectrograms ---")
all_specs = load_all_performance_specs(file_path)
print(f"Successfully loaded {len(all_specs)} spectrograms")

# Show shapes for each
print("\nSpectrogram shapes:")
for perf_name, spec in all_specs.items():
    duration = spec.shape[1] / 20
    print(f"  {perf_name}: {spec.shape} ({duration:.1f}s)")
