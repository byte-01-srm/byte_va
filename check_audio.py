import sounddevice as sd
import numpy as np

print("--- Starting Audio Check ---")
try:
    devices = sd.query_devices()
    if not devices:
        print("Error: PortAudio returned an empty list.")
    else:
        print(devices)
except Exception as e:
    print(f"Caught an error during query: {e}") 