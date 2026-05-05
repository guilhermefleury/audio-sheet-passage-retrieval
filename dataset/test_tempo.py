from audio import get_performance_list, parse_performance_name, extract_lilypond_metadata

file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"
perfs = get_performance_list(file_path)
meta = extract_lilypond_metadata(file_path)
base_bpm = meta['tempo']['bpm']

print(f"Base tempo from score: {base_bpm} BPM (quarter note)\n")
print("Performance variations:")
print(f"{'Tempo Value':<12} {'Speed Factor':<15} {'Actual BPM':<12} Instrument")
print("-" * 80)

for perf in perfs:
    info = parse_performance_name(perf)
    actual_bpm = base_bpm * info['tempo_factor']
    print(f"{info['tempo_value']:<12} {info['tempo_factor']:.2f}x{'':<11} {actual_bpm:.0f} BPM{'':<6} {info['instrument']}")
