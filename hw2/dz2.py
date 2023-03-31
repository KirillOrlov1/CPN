import matplotlib.pyplot as plt
import numpy as np


class Noise_hw:

    def __init__(self) -> None:
        self.N = 13
        self.threshold = 0.7

        self.A = 10 * self.N
        self.B = 2 * self.N
        self.fc = 2_000 * self.N
        self.fm = 150 * self.N

        self.fs = 5 * self.fc

    def make_am_signal(self, t: list[float], noise: bool = True, noise_scale: float = 0.3) -> list[float]:
        x1 = self.A * np.sin(2 * np.pi * self.fc * t)
        x2 = self.B * np.sin(2 * np.pi * self.fm * t)
        signal = (1 + x2 / self.B) * x1
        if noise:
            signal += np.random.normal(0, noise_scale * self.A, signal.shape)
        return signal

    def make_fm_signal(self, t: list[float], noise: bool = True, noise_scale: float = 0.3) -> list[float]:
        x2 = self.B * np.sin(2 * np.pi * self.fm * t)
        signal = self.A * np.sin(2 * np.pi * self.fc * t + x2)
        if noise:
            signal += np.random.normal(0, noise_scale * self.A, signal.shape)
        return signal

    def draw_plot(self, x: list[float], y: list[float], xlabel: str, ylabel: str,
                  title: str, filename: str) -> None:
        _, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, self.A + 2 * self.B)
        ax.set_title(title)

        x, y = zip(*sorted(zip(x, y)))
        ax.plot(x, y)
        plt.savefig(f"dz2_img/{filename}")

    def all_plots(self, make_signal: callable, pre: str) -> None:
        t = np.arange(15 * self.fc / self.fm) / self.fs
        wave = make_signal(t, False)
        wave_with_noise = make_signal(t, True)

        # fast FT
        freqs = np.fft.fftfreq(len(wave), 1 / self.fs)
        ffts = np.abs(np.fft.fft(wave)) / len(wave)
        ffts_with_noise = np.abs(np.fft.fft(wave_with_noise) / len(wave_with_noise))

        # spectrum treshhold
        ffts_filtered_raw = np.fft.fft(wave_with_noise) / len(wave)
        mask = np.abs(ffts_filtered_raw) < self.threshold
        ffts_filtered_raw[mask] = 0
        ffts_filtered = np.abs(ffts_filtered_raw)

        # Inverse fast FT
        wave_filtered = np.abs(np.fft.ifft(ffts_filtered_raw)) * len(wave)

        # Signal plot
        self.draw_plot(t, wave, "Time, s", "Amplitude", f"{pre} signal, without noise", f"{pre}_signal.png")
        self.draw_plot(t, wave_with_noise, "Time, s", "Amplitude", f"{pre} signal, with noise",
                       f"{pre}_signal_with_noise.png")

        # Double-sided spectrum
        self.draw_plot(freqs, ffts, "Frequency, Hz", "Amplitude",
                       f"Double-sided spectrum of {pre} signal, without noise", f"{pre}_spectrum_double_sided.png")
        self.draw_plot(freqs, ffts_with_noise, "Frequency, Hz", "Amplitude",
                       f"Double-sided spectrum of {pre} signal, with noise",
                       f"{pre}_spectrum_double_sided_with_noise.png")
        self.draw_plot(freqs, ffts_filtered, "Frequency, Hz", "Amplitude",
                       f"Double-sided spectrum of {pre} signal, after noise filtering",
                       f"{pre}_spectrum_double_sided_filtered.png")

        # Single-sided spectrum
        l = len(wave) // 2
        freqs = np.fft.fftfreq(len(wave), 1 / self.fs)[:l]
        ffts = (np.abs(np.fft.fft(wave)) / len(wave))[:l]
        ffts_with_noise = (np.abs(np.fft.fft(wave_with_noise)) / len(wave_with_noise))[:l]

        ffts_filtered_raw = np.fft.fft(wave_with_noise) / len(wave)
        mask = np.abs(ffts_filtered_raw) < self.threshold
        ffts_filtered_raw[mask] = 0
        ffts_filtered = np.abs(ffts_filtered_raw)

        self.draw_plot(freqs, ffts, "Frequency, Hz", "Amplitude",
                       f"Single-sided spectrum of {pre} signal, without noise", f"{pre}_spectrum_single_sided.png")
        self.draw_plot(freqs, ffts_with_noise, "Frequency, Hz", "Amplitude",
                       f"Single-sided spectrum of {pre} signal, with noise",
                       f"{pre}_spectrum_single_sided_with_noise.png")
        self.draw_plot(freqs, ffts_filtered, "Frequency, Hz", "Amplitude",
                       f"Single-sided spectrum of {pre} signal, after noise filtering",
                       f"{pre}_spectrum_single_sided_filtered.png")

        # Filtered signal (after inverse DFT)
        self.draw_plot(t, wave_filtered, "Time, s", "Amplitude", f"{pre} signal, after noise filtering",
                       f"{pre}_signal_filtered.png")

    def do_work(self):
        self.all_plots(self.make_am_signal, "AM")
        self.all_plots(self.make_fm_signal, "FM")


if __name__ == '__main__':
    hw = Noise_hw()
    hw.do_work()
