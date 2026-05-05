from sheet import get_bars_in_song

# Test on Bach Invention 08 (has 2 pages)
bars = get_bars_in_song("../data/msmd/BachJS__BWV779__bach-invention-08")

print(f"Total systems in piece: {len(bars)}")
print(f"Bars per system: {bars}")
print(f"Total bars in piece: {sum(bars)}")

# Show breakdown by page
page1_systems = 7  # We know from previous testing
print(f"\nPage 1 systems (first {page1_systems}): {bars[:page1_systems]}")
print(f"Page 2 systems (remaining): {bars[page1_systems:]}")
