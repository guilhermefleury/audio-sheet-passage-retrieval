from audio import calculate_system_durations, extract_lilypond_metadata

file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"

# Test with base tempo (90 BPM)
print("=== Base Tempo (90 BPM) ===")
metadata = extract_lilypond_metadata(file_path)
print(f"Time signature: {metadata['time_signature'][0]}/{metadata['time_signature'][1]}")
print(f"Tempo: {metadata['tempo']['bpm']} BPM\n")

timings = calculate_system_durations(file_path)

print(f"{'System':<8} {'Bars':<6} {'Start':<10} {'End':<10} {'Duration':<10}")
print("-" * 50)
for t in timings:
    print(f"{t['system_index']:<8} {t['bars']:<6} {t['start_time']:<10.2f} {t['end_time']:<10.2f} {t['duration']:<10.2f}")

print(f"\nTotal duration: {timings[-1]['end_time']:.2f} seconds")

# Test with a faster performance (tempo-500 = 2x faster = 180 BPM)
print("\n\n=== Fast Performance (tempo-500 = 180 BPM) ===")
timings_fast = calculate_system_durations(file_path, "BachJS__BWV779__bach-invention-08_tempo-500_grand-piano-YDP-20160804")

print(f"{'System':<8} {'Bars':<6} {'Start':<10} {'End':<10} {'Duration':<10}")
print("-" * 50)
for t in timings_fast[:5]:  # Show first 5 systems
    print(f"{t['system_index']:<8} {t['bars']:<6} {t['start_time']:<10.2f} {t['end_time']:<10.2f} {t['duration']:<10.2f}")

print(f"...\nTotal duration: {timings_fast[-1]['end_time']:.2f} seconds")
print(f"Speedup: {timings[-1]['end_time'] / timings_fast[-1]['end_time']:.2f}x")
