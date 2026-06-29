#!/usr/bin/env python3
"""Quick audio level test for debugging STT"""
import subprocess
import numpy as np
import tempfile
import os

print("Testing audio levels from ReSpeaker...")
print("Speak into the microphone!")
print("-" * 50)

for i in range(5):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        # Record 2 seconds
        cmd = [
            'arecord', '-D', 'plughw:1,0',
            '-d', '2',
            '-f', 'S16_LE', '-r', '16000', '-c', '1', '-q',
            tmp_filename
        ]

        print(f"\nRecording {i+1}/5...")
        subprocess.run(cmd, capture_output=True)

        # Check energy
        with open(tmp_filename, 'rb') as f:
            f.seek(44)  # Skip WAV header
            audio_data = np.frombuffer(f.read(), dtype=np.int16)
            energy = np.abs(audio_data).mean() if len(audio_data) > 0 else 0
            max_amplitude = np.abs(audio_data).max() if len(audio_data) > 0 else 0

        print(f"  Energy level: {energy:.1f}")
        print(f"  Max amplitude: {max_amplitude}")
        print(f"  Status: {'SPEECH DETECTED' if energy > 100 else 'silence/too quiet'}")

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

print("\n" + "-" * 50)
print("Test complete!")
print("If energy levels are consistently below 100, the threshold is too high")
print("or the microphone volume needs adjustment.")
