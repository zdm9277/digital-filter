"""
CSE 3313 Team Project - Team 10
IIR low-pass filter: Bilinear Transformation + Least p-norm design

written by Zachary Montgomery

Steps
  1. Pre-warp the digital band edges to analog:  W = 2*fs*tan(w/2)
  2. Design an analog filter H(s) by least p-norm optimization
  3. Convert H(s) -> H(z) with the bilinear transform
  4. Increase the order until the specs are met
  5. Filter a noisy voice recording and plot the results

Usage
  python team10_lpnorm_bilinear.py                          (uses team10_noisy_48k.wav)
  python team10_lpnorm_bilinear.py noisy.wav                (any noisy .wav file)
  python team10_lpnorm_bilinear.py clean.wav --add-noise    (adds our test noise first)

Any input that is not 48 kHz is resampled to 48 kHz before filtering.
"""

import sys
import numpy as np
import scipy.signal as sig
from scipy.optimize import minimize
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")                       # save plots straight to PNG files, no pop-up windows
import matplotlib.pyplot as plt

FS = 48000                   # required sampling rate (Hz); other inputs get resampled
DEFAULT_WAV = "team10_noisy_48k.wav"   # noisy input submitted with the code

# ---------------- Specifications ----------------
FP = 2000     # passband edge (Hz)
FST = 2500    # stopband edge (Hz)
RP = 1.0      # max passband ripple (dB)
AS = 40.0     # min stopband attenuation (dB)

# The optimizer works with plain magnitudes, not dB, so convert the specs.
#   passband: |H| has to stay between 1 and 1 - DP
#   stopband: |H| has to stay below DS
DP = 1 - 10 ** (-RP / 20)   # passband ripple as a linear amount (~0.109)
DS = 10 ** (-AS / 20)       # stopband limit as a linear amount (0.01)


# ---------------- Analog filter H(s) ----------------
# For order N the filter has:
#   N//2 complex pole pairs     -sigma +/- j*omega
#   N//2 zero pairs on the jW axis    +/- j*wz   (they make the stopband notches)
#   one extra real pole -r if N is odd
#   a gain k
# All of these are packed into one parameter list so the optimizer can tune them.
# abs() keeps every pole in the left half-plane, so H(s) is always stable.
#
# Layout of params (m = N//2):
#   [ sigma_1..sigma_m | omega_1..omega_m | wz_1..wz_m | r (odd N only) | k ]
#
# Note: everything here is in *normalized* frequency, where the passband edge
# is at W = 1 rad/s. Keeping the numbers near 1 makes the optimizer much happier
# than working with values in the tens of thousands. We scale back up at the end.

def params_to_zpk(params, N):
    """Unpack the parameter list into zeros, poles and gain."""
    m = N // 2
    sigma = np.abs(params[0:m])                  # real parts (forced positive -> poles in LHP)
    omega = params[m:2 * m]                      # imaginary parts of the poles
    wz = params[2 * m:3 * m]                     # where the notches (zeros) sit on the jW axis
    k = params[-1]                               # overall gain

    # Each pole/zero gets its complex conjugate so H(s) has real coefficients
    poles = np.concatenate([-sigma + 1j * omega, -sigma - 1j * omega])
    zeros = np.concatenate([1j * wz, -1j * wz])
    if N % 2 == 1:
        poles = np.append(poles, -abs(params[3 * m]))   # the extra real pole for odd N
    return zeros, poles, k


def analog_magnitude(params, N, W):
    """|H(jW)| at the analog frequencies W."""
    zeros, poles, k = params_to_zpk(params, N)
    _, h = sig.freqs_zpk(zeros, poles, k, worN=W)
    return np.abs(h)


def starting_guess(N, ws):
    """Start from a Butterworth filter, with zeros spread over the stopband.

    The optimizer needs somewhere reasonable to start. A Butterworth low-pass
    is already roughly the right shape, so we just let the optimizer improve it.
    ws is the (normalized) analog stopband edge.
    """
    _, poles, _ = sig.buttap(N)                  # Butterworth poles, cutoff at 1 rad/s
    poles = poles * (1 + ws) / 2                 # cutoff between the band edges
    upper = poles[poles.imag > 1e-9]             # one pole from each pair
    m = N // 2
    # sigma's, omega's, then zeros spaced from just past the stopband edge out to 4x it
    params = list(-upper.real) + list(upper.imag) + list(ws * np.geomspace(1.05, 4, m))
    if N % 2 == 1:
        params.append(-poles[np.abs(poles.imag) < 1e-9].real[0])   # the real pole
    params.append(1.0)                           # temporary gain
    params = np.array(params)
    params[-1] = 1 / analog_magnitude(params, N, [0])[0]   # make |H(0)| = 1
    return params


# ---------------- Least p-norm design ----------------
# Idea: pick a bunch of frequencies in the passband and stopband, measure how far
# |H| is from where we want it, and have the optimizer shrink that error.
#
# How we "add up" the error is the p-norm:
#   p = 2    -> least squares (average error is small, but peaks can stick out)
#   p -> inf -> minimax (only the single worst error matters)
# The specs are about the WORST point in each band, so we really want a large p.
# Jumping straight to p = 128 tends to get stuck, so we start at p = 2 and
# double p each round, using the last answer as the next starting point.

def design_filter(N, fs):
    """Design an order-N digital low-pass. Returns a dict with all the pieces."""

    # Step 1: band edges in digital frequency (rad/sample), then pre-warp to analog (rad/s).
    # The bilinear transform squishes the analog frequency axis, so we pre-warp
    # so that the edges land exactly at FP and FST after the transform.
    wp = 2 * np.pi * FP / fs
    ws = 2 * np.pi * FST / fs
    Wp = 2 * fs * np.tan(wp / 2)
    Ws = 2 * fs * np.tan(ws / 2)

    # Frequency grid: 400 points in the passband, 400 in the stopband.
    # (Nothing in the transition band; we don't care what happens there.)
    # Pick the points in digital frequency, pre-warp each one, then divide by Wp
    # so the passband edge sits at 1.
    w_pass = np.linspace(0, wp, 400)
    w_stop = np.linspace(ws, 0.999 * np.pi, 400)   # stop just short of pi, since tan(pi/2) blows up
    W = 2 * fs * np.tan(np.concatenate([w_pass, w_stop]) / 2) / Wp

    # Target response and weights.
    #   passband target is the middle of the allowed band (1 - DP/2), so |H| can
    #   wiggle DP/2 up or down and still pass.
    #   stopband target is 0.
    # The weights scale each band's error so that "right at the spec limit"
    # counts as an error of 1 in BOTH bands. That way neither band gets favored.
    desired = np.concatenate([np.full(400, 1 - DP / 2), np.zeros(400)])
    weight = np.concatenate([np.full(400, 2 / DP), np.full(400, 1 / DS)])

    def p_norm_error(params, p):
        """Weighted p-norm of the error. This is what the optimizer minimizes."""
        e = weight * np.abs(analog_magnitude(params, N, W) - desired)
        # Raising numbers to the 128th power can overflow, so pull out the
        # biggest error first. Mathematically it's the same p-norm.
        big = e.max()
        return big * np.mean((e / big) ** p) ** (1 / p)

    # Step 2: optimize, slowly cranking up p
    params = starting_guess(N, Ws / Wp)
    history = {}                                 # keep each p's result for the "effect of p" plot
    for p in [2, 4, 8, 16, 32, 64, 128]:
        params = minimize(p_norm_error, params, args=(p,), method="BFGS").x
        history[p] = params.copy()

    # Undo the normalization: scaling s by Wp moves every pole and zero by Wp.
    # The gain also has to change by Wp^(#poles - #zeros) or the overall level shifts.
    zeros, poles, k = params_to_zpk(params, N)
    k = k * Wp ** (len(poles) - len(zeros))

    # Step 3: bilinear transform, s = 2*fs*(z - 1)/(z + 1), maps H(s) -> H(z)
    zd, pd, kd = sig.bilinear_zpk(zeros * Wp, poles * Wp, k, fs)

    # Final touch: rescale the gain so the passband peaks at exactly 1 (0 dB)
    _, h = sig.freqz_zpk(zd, pd, kd, worN=np.linspace(0, wp, 2000))
    kd = kd / np.abs(h).max()

    return dict(N=N, Wp=Wp, history=history,
                analog=(zeros * Wp, poles * Wp, k), digital=(zd, pd, kd))


def measure(sos, fs):
    """Passband ripple (dB) and stopband attenuation (dB) of the digital filter."""
    f, h = sig.sosfreqz(sos, worN=8192, fs=fs)
    mag = np.abs(h)
    passband, stopband = mag[f <= FP], mag[f >= FST]
    ripple = 20 * np.log10(passband.max() / passband.min())     # how much the passband wobbles
    atten = -20 * np.log10(stopband.max() / passband.max())     # worst stopband point vs. passband
    return ripple, atten


def find_min_order(fs, max_order=12):
    """Step 4: try N = 2, 3, 4, ... and return the first design that meets every spec."""
    for N in range(2, max_order + 1):
        d = design_filter(N, fs)
        sos = sig.zpk2sos(*d["digital"])
        ripple, atten = measure(sos, fs)
        stable = np.all(np.abs(d["digital"][1]) < 1)            # digital filter is stable if all poles are inside the unit circle
        print(f"  N={N:2d}: ripple {ripple:5.2f} dB, atten {atten:5.1f} dB, stable={stable}")
        if ripple <= RP and atten >= AS and stable:
            d.update(sos=sos, ripple=ripple, atten=atten)
            return d
    raise RuntimeError("Specs not met up to max_order")


# ---------------- Plots ----------------
def shade_bands(ax, fs):
    """Color in the passband / transition / stopband so the plots are easy to read."""
    ax.axvspan(0, FP, color="g", alpha=0.12, label="Pass-band")
    ax.axvspan(FP, FST, color="orange", alpha=0.15, label="Transition region")
    ax.axvspan(FST, fs / 2, color="r", alpha=0.10, label="Stop-band")
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(alpha=0.3)


def to_db(x):
    """Magnitude -> dB. The floor at 1e-8 avoids log(0) right at the notches."""
    return 20 * np.log10(np.maximum(x, 1e-8))


def plot_response(d, fs):
    """Fig 1: frequency response, linear on top and dB on the bottom, with spec lines."""
    f, h = sig.sosfreqz(d["sos"], worN=8192, fs=fs)
    mag = np.abs(h)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7))

    # Linear scale: dashed lines show the allowed passband range and stopband ceiling
    shade_bands(top, fs)
    top.plot(f, mag, "b")
    top.hlines([1, 1 - DP], 0, FP, "k", "--", lw=0.8)
    top.hlines(DS, FST, fs / 2, "k", "--", lw=0.8)
    top.set(title=f"Team 10 - Bilinear + least p-norm IIR low-pass (N={d['N']}, fs={fs} Hz)",
            ylabel="|H|")
    top.legend(loc="upper right")

    # dB scale: same spec lines, now at -RP and -AS
    shade_bands(bottom, fs)
    bottom.plot(f, to_db(mag), "b")
    bottom.hlines(-RP, 0, FP, "k", "--", lw=0.8)
    bottom.hlines(-AS, FST, fs / 2, "k", "--", lw=0.8)
    bottom.set(ylabel="Magnitude (dB)", ylim=(-100, 5))

    fig.tight_layout()
    fig.savefig("fig1_filter_response.png", dpi=150)


def plot_p_effect(d, fs):
    """Fig 2: how the response changes as p goes up (uses the saved history)."""
    # Map each plotted Hz value to the matching normalized analog frequency,
    # so the x-axis lines up with the digital filter's frequencies.
    f = np.linspace(0, 0.999 * fs / 2, 2000)
    W = 2 * fs * np.tan(np.pi * f / fs) / d["Wp"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for p in [2, 8, 32, 128]:
        ax.plot(f, to_db(analog_magnitude(d["history"][p], d["N"], W)), label=f"p = {p}")
    ax.axvline(FP, color="k", ls=":")
    ax.axvline(FST, color="k", ls=":")
    ax.axhline(-AS, color="k", ls="--", lw=0.8)
    ax.set(title="Effect of p: larger p -> smaller peak error (equiripple)",
           xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, fs / 2), ylim=(-90, 5))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_pnorm_effect.png", dpi=150)


def plot_pole_zero(d):
    """Fig 3: pole-zero plots before (s-plane) and after (z-plane) the bilinear transform.

    Stable analog: poles in the left half-plane.
    Stable digital: poles inside the unit circle.
    The bilinear transform maps one onto the other.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (z, p, _), title in [(left, d["analog"], "Analog prototype H(s)"),
                                 (right, d["digital"], "Digital filter H(z) after bilinear transform")]:
        ax.plot(z.real, z.imag, "o", mfc="none", ms=8, label="zeros")
        ax.plot(p.real, p.imag, "x", ms=8, label="poles")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    left.axvline(0, color="k", lw=0.6)                     # the jW axis
    left.set(xlabel="σ", ylabel="jΩ")
    t = np.linspace(0, 2 * np.pi, 400)
    right.plot(np.cos(t), np.sin(t), "k", lw=0.6)          # unit circle
    right.set(xlabel="Re", ylabel="Im", aspect="equal")
    fig.tight_layout()
    fig.savefig("fig3_pole_zero.png", dpi=150)


def plot_signals(x, y, fs, sos):
    """Figs 4 and 5: the audio before/after filtering, in time and in frequency."""
    # Time domain
    t = np.arange(len(x)) / fs                   # sample index -> seconds
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(t, x, color="tab:orange", lw=0.5)
    ax[0].set_title("Audio Before Filtering")
    ax[1].plot(t, y, color="tab:blue", lw=0.5)
    ax[1].set_title("Audio After Filtering")
    ax[1].set_xlabel("Time (s)")
    for a in ax:
        a.set_ylim(-1, 1)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig4_time_domain.png", dpi=150)

    # Frequency domain: DFT before/after with |H| on top
    # fftshift moves 0 Hz to the middle so the plot runs from -fs/2 to +fs/2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
    X = np.abs(np.fft.fftshift(np.fft.fft(x)))
    Y = np.abs(np.fft.fftshift(np.fft.fft(y)))
    # |H| is symmetric for a real filter, so |H(-f)| = |H(f)|; just evaluate at |f|
    _, h = sig.sosfreqz(sos, worN=np.abs(freqs), fs=fs)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for a, spectrum, title in [(ax[0], X, "DFT Before Filtering"), (ax[1], Y, "DFT After Filtering")]:
        # both spectra divided by the same number so they're directly comparable
        a.plot(freqs, spectrum / X.max(), lw=0.5, label="|DFT| (normalized)")
        a.plot(freqs, np.abs(h), "r", label="|H|")
        a.set(title=title, ylim=(0, 1.05))
        a.grid(alpha=0.3)
    ax[0].legend(loc="upper right")
    ax[1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig("fig5_dft.png", dpi=150)


# ---------------- Audio ----------------
def load_wav(path):
    """Read a .wav file, convert to mono, resample to 48 kHz, and scale to [-1, 1]."""
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)                       # stereo -> mono
    x = x.astype(float)
    if fs != FS:
        # Resample to 48 kHz (same idea as MATLAB resample): up by FS/g, down by fs/g
        g = np.gcd(int(fs), FS)
        x = sig.resample_poly(x, FS // g, int(fs) // g)
        print(f"Resampled {path} from {fs} Hz to {FS} Hz")
        fs = FS
    return fs, x / np.max(np.abs(x))             # scale to [-1, 1]


def add_noise(x, fs):
    """Add hiss above the stopband edge (high-passed white noise).

    The noise only lives in the stopband, so a working filter should
    remove almost all of it and leave the voice alone.
    """
    noise = np.random.default_rng(0).standard_normal(len(x))   # fixed seed = same noise every run
    highpass = sig.butter(8, 1.05 * FST, "highpass", fs=fs, output="sos")
    noise = sig.sosfilt(highpass, noise)
    y = x + 0.5 * noise / np.max(np.abs(noise))
    return y / np.max(np.abs(y))                 # re-scale so it doesn't clip


def save_wav(path, fs, x):
    """Save a [-1, 1] signal as a 16-bit .wav file."""
    wavfile.write(path, fs, (x * 32767).astype(np.int16))


# ---------------- Main ----------------
def main():
    # Load the noisy audio (default: the noisy 48 kHz file submitted with this code).
    # Only add our test noise if asked to (for starting from a clean recording).
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = files[0] if files else DEFAULT_WAV
    fs, x = load_wav(path)
    if FST >= fs / 2:
        sys.exit(f"Stopband edge {FST} Hz must be below fs/2 = {fs / 2} Hz")   # Nyquist check
    if "--add-noise" in sys.argv:
        x = add_noise(x, fs)

    # Design the filter
    print(f"Specs: fp={FP} Hz, fst={FST} Hz, Rp={RP} dB, As={AS} dB, fs={fs} Hz")
    print("Searching for minimum order:")
    d = find_min_order(fs)

    # Print the results, including the difference-equation coefficients b (numerator) and a (denominator)
    b, a = sig.zpk2tf(*d["digital"])
    print(f"\nFinal design: order N = {d['N']}")
    print(f"  passband ripple = {d['ripple']:.3f} dB (spec <= {RP})")
    print(f"  stopband atten. = {d['atten']:.2f} dB (spec >= {AS})")
    print(f"  max |pole|      = {np.abs(d['digital'][1]).max():.4f} (< 1 means stable)")
    print("  b =", np.round(b.real, 6))
    print("  a =", np.round(a.real, 6))

    # Step 5: filter the audio.
    # We use second-order sections (a chain of small 2nd-order filters) instead of
    # one big b/a filter. High-order b/a coefficients are sensitive to rounding
    # and can go unstable; the SOS version doesn't have that problem.
    y = sig.sosfilt(d["sos"], x)
    y = y / np.max(np.abs(y))
    save_wav("noisy_input.wav", fs, x)
    save_wav("filtered_output.wav", fs, y)

    # Make all the figures
    plot_response(d, fs)
    plot_p_effect(d, fs)
    plot_pole_zero(d)
    plot_signals(x, y, fs, d["sos"])
    print("\nSaved: fig1..fig5 PNGs, noisy_input.wav, filtered_output.wav")


if __name__ == "__main__":
    main()
