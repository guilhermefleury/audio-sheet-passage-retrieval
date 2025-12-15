import cv2
import numpy as np
from pathlib import Path
import re

def get_performance_list(file_path):
    """
    Returns a list of all performance folder names for a given piece.
    
    Args:
        file_path: Path to the MSMD piece directory
    
    Returns:
        List of performance folder names
    """
    performances_dir = Path(file_path) / "performances"
    
    if not performances_dir.exists():
        return []
    
    # Get all subdirectories in performances folder
    performance_folders = [f.name for f in performances_dir.iterdir() if f.is_dir()]
    return sorted(performance_folders)

def parse_performance_name(performance_name, base_tempo_value=1000):
    """
    Extracts tempo factor and instrument from performance folder name.
    
    Args:
        performance_name: e.g., 'piece_tempo-500_grand-piano-YDP-20160804'
        base_tempo_value: The tempo value representing 1.0x speed (default: 1000)
    
    Returns:
        Dictionary with:
            - 'tempo_factor': speed multiplier (e.g., tempo-500 = 2.0x faster, tempo-2000 = 0.5x slower)
            - 'tempo_value': the raw tempo number from folder name
            - 'instrument': instrument identifier
    
    Note: Smaller tempo values = faster playback (inversely proportional)
    """
    info = {'tempo_factor': None, 'tempo_value': None, 'instrument': None}
    
    # Extract tempo: tempo-500 is faster (1000/500 = 2.0x), tempo-2000 is slower (1000/2000 = 0.5x)
    tempo_match = re.search(r'tempo-(\d+)', performance_name)
    if tempo_match:
        tempo_value = int(tempo_match.group(1))
        info['tempo_value'] = tempo_value
        info['tempo_factor'] = base_tempo_value / tempo_value  # Inverse relationship
    
    # Extract instrument (everything after tempo-XXXX_)
    instrument_match = re.search(r'tempo-\d+_(.+)', performance_name)
    if instrument_match:
        info['instrument'] = instrument_match.group(1)
    
    return info

def load_performance_spec(file_path, performance_name):
    """
    Loads the spectrogram for a specific performance.
    
    Args:
        file_path: Path to the MSMD piece directory
        performance_name: Name of the performance folder (e.g., 'piece_tempo-1000_acoustic_piano_imis_1')
    
    Returns:
        NumPy array containing the spectrogram
    """
    perf_dir = Path(file_path) / "performances" / performance_name / "features"
    
    # Find the .flac_spec.npy file
    spec_files = list(perf_dir.glob("*.flac_spec.npy"))
    if not spec_files:
        raise FileNotFoundError(f"No spectrogram file found in {perf_dir}")
    
    return np.load(spec_files[0])

def load_all_performance_specs(file_path):
    """
    Loads spectrograms for all performances of a piece.
    
    Args:
        file_path: Path to the MSMD piece directory
    
    Returns:
        Dictionary mapping performance names to their spectrograms
    """
    specs = {}
    performance_list = get_performance_list(file_path)
    
    for perf_name in performance_list:
        try:
            specs[perf_name] = load_performance_spec(file_path, perf_name)
        except Exception as e:
            print(f"Warning: Failed to load {perf_name}: {e}")
    
    return specs


def extract_lilypond_metadata(file_path):
    """
    Extracts metadata from LilyPond (.ly) file.
    
    Args:
        file_path: Path to the MSMD piece directory
    
    Returns:
        Dictionary with keys:
            - 'time_signature': tuple (numerator, denominator) or None
            - 'tempo': dict with 'bpm' and 'beat_unit', or None
    """
    piece_name = Path(file_path).name
    
    # Try .norm.ly first, then .ly
    ly_files = [
        Path(file_path) / f"{piece_name}.norm.ly",
        Path(file_path) / f"{piece_name}.ly"
    ]
    
    ly_file = None
    for f in ly_files:
        if f.exists():
            ly_file = f
            break
    
    if not ly_file:
        return {'time_signature': None, 'tempo': None}
    
    metadata = {'time_signature': None, 'tempo': None}
    
    with open(ly_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract time signature: \time 3/4
    time_match = re.search(r'\\time\s+(\d+)/(\d+)', content)
    if time_match:
        metadata['time_signature'] = (int(time_match.group(1)), int(time_match.group(2)))
    
    # Extract tempo - try multiple formats:
    # 1. \tempo 4 = 90 (most common in MIDI blocks)
    tempo_match = re.search(r'\\tempo\s+(\d+)\s*=\s*(\d+)', content)
    if tempo_match:
        beat_unit = int(tempo_match.group(1))  # e.g., 4 means quarter note
        bpm = int(tempo_match.group(2))
        metadata['tempo'] = {'bpm': bpm, 'beat_unit': beat_unit}
    else:
        # 2. tempoWholesPerMinute = #(ly:make-moment 90 4)
        # This means 90 beats of 1/4 notes per minute = 90 BPM
        tempo_match = re.search(r'tempoWholesPerMinute\s*=\s*#\(ly:make-moment\s+(\d+)\s+(\d+)\)', content)
        if tempo_match:
            bpm = int(tempo_match.group(1))
            beat_unit = int(tempo_match.group(2))
            metadata['tempo'] = {'bpm': bpm, 'beat_unit': beat_unit}
    
    return metadata

def calculate_system_durations(file_path, performance_name=None):
    """
    Calculates the start time and duration of each system based on musical timing.
    
    Args:
        file_path: Path to the MSMD piece directory
        performance_name: Optional performance name to get tempo factor. If None, uses base tempo.
    
    Returns:
        List of dictionaries, one per system, with keys:
            - 'system_index': system number (0-based)
            - 'bars': number of bars in this system
            - 'start_time': start time in seconds
            - 'end_time': end time in seconds
            - 'duration': duration in seconds
    """
    # Import here to avoid circular dependency
    import sys
    sys.path.append(str(Path(__file__).parent))
    from sheet import get_bars_in_song
    
    # Get musical metadata
    metadata = extract_lilypond_metadata(file_path)
    if not metadata['time_signature'] or not metadata['tempo']:
        raise ValueError(f"Missing time signature or tempo in {file_path}")
    
    beats_per_bar, beat_unit = metadata['time_signature']  # e.g., (3, 4) = 3/4 time
    base_bpm = metadata['tempo']['bpm']
    
    # Apply tempo factor if performance specified
    tempo_factor = 1.0
    if performance_name:
        perf_info = parse_performance_name(performance_name)
        tempo_factor = perf_info['tempo_factor']
    
    actual_bpm = base_bpm * tempo_factor
    
    # Calculate seconds per bar
    # beats_per_bar beats × (60 seconds / actual_bpm beats) = seconds per bar
    seconds_per_bar = (beats_per_bar * 60.0) / actual_bpm
    
    # Get bars per system
    bars_per_system = get_bars_in_song(file_path)
    
    # Calculate timing for each system
    system_timings = []
    current_time = 0.0
    
    for system_idx, num_bars in enumerate(bars_per_system):
        duration = num_bars * seconds_per_bar
        system_timings.append({
            'system_index': system_idx,
            'bars': num_bars,
            'start_time': current_time,
            'end_time': current_time + duration,
            'duration': duration
        })
        current_time += duration
    
    return system_timings

def slice_spectrogram_by_systems(file_path, performance_name, fps=20):
    """
    Slices a performance spectrogram into system-aligned segments.
    
    Args:
        file_path: Path to the MSMD piece directory
        performance_name: Name of the performance to load
        fps: Frames per second of the spectrogram (default: 20)
    
    Returns:
        List of numpy arrays, one spectrogram per system
    """
    # Load the full spectrogram
    spec = load_performance_spec(file_path, performance_name)
    
    # Get system timings
    timings = calculate_system_durations(file_path, performance_name)
    
    # Slice spectrogram by time
    system_specs = []
    for timing in timings:
        start_frame = int(timing['start_time'] * fps)
        end_frame = int(timing['end_time'] * fps)
        
        # Extract this system's spectrogram
        system_spec = spec[:, start_frame:end_frame]
        system_specs.append(system_spec)
    
    return system_specs

def system_spec_slicer(system_spec, window_frames=20, stride_frames=None):
    """
    Slices a system spectrogram into 1-second windows (92 × 20 frames).
    The final window is anchored at the end of the system (counting 1s back from the end).
    All windows stay within system boundaries.
    
    Args:
        system_spec: System spectrogram array (freq_bins, frames)
        window_frames: Number of frames per window (default: 20 = 1 second at 20 fps)
        stride_frames: Stride between windows. If None, uses window_frames (no overlap)
    
    Returns:
        List of spectrogram snippets, each (freq_bins, window_frames)
    """
    if stride_frames is None:
        stride_frames = window_frames
    
    if system_spec.shape[1] < window_frames:
        # System is shorter than window size, return empty list or pad
        return []
    
    system_sequence = []
    frame_iterator = 0
    
    # Slide window across the system
    while frame_iterator + window_frames <= system_spec.shape[1]:
        snippet = system_spec[:, frame_iterator:frame_iterator + window_frames]
        system_sequence.append(snippet)
        frame_iterator += stride_frames
    
    # Add final window anchored at the end (counting back from end)
    # Only add if it doesn't duplicate the last window we just added
    last_window_start = system_spec.shape[1] - window_frames
    last_added_start = len(system_sequence) * stride_frames - stride_frames if system_sequence else -1
    
    if last_window_start > last_added_start:
        snippet = system_spec[:, last_window_start:system_spec.shape[1]]
        system_sequence.append(snippet)
    
    return system_sequence

def get_spec_systems(file_path, performance_name, window_frames=20, stride_frames=None, fps=20):
    """
    Gets system spectrograms and their 1-second sliced sequences.
    Similar to get_sheet_systems but for audio spectrograms.
    
    Args:
        file_path: Path to the MSMD piece directory
        performance_name: Name of the performance to load
        window_frames: Frames per window (default: 20 = 1 second)
        stride_frames: Stride between windows. If None, uses window_frames
        fps: Frames per second (default: 20)
    
    Returns:
        Tuple of (systems, systems_sliced) where:
            - systems: List of full system spectrograms
            - systems_sliced: List of lists, each containing 1-second slices for that system
    """
    # Get full system spectrograms
    systems = slice_spectrogram_by_systems(file_path, performance_name, fps)
    
    # Slice each system into 1-second windows
    systems_sliced = []
    for system_spec in systems:
        slices = system_spec_slicer(system_spec, window_frames, stride_frames)
        systems_sliced.append(slices)
    
    return systems, systems_sliced

