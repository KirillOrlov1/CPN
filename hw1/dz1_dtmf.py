import pyaudio as pa
import numpy as np
import matplotlib.pyplot as plt
import time


class Sound:

    def __init__(self) -> None:
        self.freqs = {
            "1": [697, 1209],
            "2": [697, 1336],
            "3": [697, 1477],
            "4": [770, 1209],
            "5": [770, 1336],
            "6": [770, 1477],
            "7": [852, 1209],
            "8": [852, 1336],
            "9": [852, 1477],
            "0": [941, 1336],
            "a": [697, 1633],
            "b": [770, 1633],
            "c": [852, 1633],
            "d": [941, 1633],
            "*": [941, 1209],
            "#": [941, 1477],
        }

        self.fs = 10_000
        self.duration = 0.5
        self.p = pa.PyAudio()
        self.stream = self.p.open(format=pa.paFloat32, channels=1, rate=self.fs, output=True)

    def play(self, s: str) -> None:
        for ch in s:
            freq = self.freqs[ch]
            samples = np.mean(
                np.sin(np.outer(freq, 2 * np.pi * np.arange(self.fs * self.duration) / self.fs)),
                axis=0).astype(np.float32).tobytes()
            self.stream.write(samples)
            time.sleep(0.3)

        for ch in s:
            freq = self.freqs[ch]
            x = np.linspace(0, 0.005, 100_000)
            y = np.mean(np.sin(np.outer(freq, 2 * np.pi * x)), axis=0)
            fig, ax = plt.subplots()
            ax.set_xlim(0, 0.005)
            ax.set_ylim(-1, 1)
            ax.set_xlabel("Time, s")
            ax.set_ylabel("Amplitudes")
            ax.set_title(f"{s} - {freq} Hz")
            ax.plot(x, y)
            plt.savefig(f"dz1_graphs/{s} - {freq} Hz.png")


DATA = set('1234567890abcd*#')

if __name__ == '__main__':
    x = input('input: ')

    if not set(x).issubset(DATA):
        raise 'input error'

    sound = Sound()
    sound.play(x)
