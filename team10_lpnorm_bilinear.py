"""CSE 3313 Team 10: corrected LPF for the new 48 kHz voice recording.

Based on Zachary Montgomery's supplied project implementation.
Method: analog IIR least p-norm optimization, then bilinear transformation.
Butterworth poles are only an initial guess, not the final design algorithm.

Run from the extracted folder:
    python team10_lpnorm_bilinear_fixed.py
    python team10_lpnorm_bilinear_fixed.py --no-add-noise
    python team10_lpnorm_bilinear_fixed.py another.wav --output-dir other_results

The default demonstration adds a known 23.05 kHz tone to the supplied recording.
Use --no-add-noise to process only the noise already in a recording.
Requires numpy, scipy and matplotlib.

DOCUMENTED BAND-EDGE ASSUMPTION:
The assignment says cutoff = 22 kHz without specifying both band edges.
This implementation treats 22 kHz as the passband edge and chooses 23 kHz
as the stopband edge. It does not impose a -3 dB point at exactly 22 kHz.
Other targets: sampling 48 kHz, order at least 20, ripple <=1 dB,
stopband attenuation >=17 dB. The implemented order is exactly 20.
"""

from pathlib import Path
from math import gcd
import argparse
import csv
import json
import numpy as np
from scipy import signal as sig, optimize as opt
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')  # Save all figures to PNG; works without a display.
import matplotlib.pyplot as plt

# 1. Project specifications and explicitly chosen test settings.
DEFAULT_WAV = 'team10_noisy_48k.wav'
FS = 48_000
FP = 22_000.0
FST = 23_000.0
RP = 1.0
AS = 17.0
ORDER = 20
P_STEPS = (2, 4, 8, 16, 32, 64, 128)
NOISE_HZ = 23_050.0
NOISE_PEAK = 0.20
SOURCE_PEAK = 0.65


def validate_settings():
    if FS != 48_000:
        raise ValueError('This assignment requires FS = 48000.')
    if not 0 < FP < FST < FS / 2:
        raise ValueError('Require 0 < passband edge < stopband edge < Nyquist.')
    if ORDER < 20 or ORDER % 2:
        raise ValueError('This implementation uses an even order of at least 20.')
    if not FST < NOISE_HZ < FS / 2:
        raise ValueError('Test noise must be inside the stopband and below Nyquist.')


# 2. Stable analog parameterization, adapted from the original implementation.
# Ten pole pairs give order 20. Positive damping makes all analog poles stable.
def unpack(theta):
    m = ORDER // 2
    damping, pole_w, zero_w = np.exp(theta[:3*m].reshape(3, m))
    poles = np.r_[-damping + 1j*pole_w, -damping - 1j*pole_w]
    zeros = np.r_[1j*zero_w, -1j*zero_w]
    return zeros, poles, float(np.exp(theta[-1]))


def analog_magnitude_and_jacobian(theta, frequencies):
    """Magnitude and its derivatives; log products avoid huge polynomial values.

    Frequencies are analog rad/s normalized to the prewarped passband edge.
    Derivatives let optimization run without many finite-difference evaluations.
    """
    m = ORDER // 2
    a, b, c = np.exp(theta[:3*m].reshape(3, m))
    s = 1j * np.asarray(frequencies)[:, None]
    v, u, q, r = s+a-1j*b, s+a+1j*b, s-1j*c, s+1j*c
    log_mag = theta[-1] + np.sum(
        np.log(np.maximum(abs(q), 1e-150))
        + np.log(np.maximum(abs(r), 1e-150))
        - np.log(abs(v)) - np.log(abs(u)), axis=1)
    mag = np.exp(log_mag)
    jac = np.c_[
        -np.real(a/v + a/u),
        np.real(1j*b/v - 1j*b/u),
        np.real(-1j*c/(q+1e-150) + 1j*c/(r+1e-150)),
        np.ones(len(frequencies))] * mag[:, None]
    return mag, jac


def initial_guess(normalized_stop):
    """Butterworth pole locations are ONLY an optimization starting point.

    Stopband zeros and every pole and gain are then optimized by least p-norm.
    The final filter is not simply scipy.signal.butter().
    """
    _, poles, _ = sig.buttap(ORDER)
    poles *= (1 + normalized_stop) / 2
    pc = poles[poles.imag > 1e-9]
    zero_w = normalized_stop * np.geomspace(1.05, 4, ORDER//2)
    theta = np.r_[np.log(-pc.real), np.log(pc.imag), np.log(zero_w), 0.0]
    theta[-1] = np.sum(np.log(abs(poles))) - 2*np.sum(np.log(zero_w))
    return theta


def check_specs(zpk):
    """Dense independent verification, including DC, exact band edges, Nyquist.

    This is a numerical grid check, not a mathematical proof over all frequencies.
    """
    sos = sig.zpk2sos(*zpk)
    f = np.unique(np.r_[np.linspace(0, FS/2, 131073), FP, FST])
    _, h = sig.sosfreqz(sos, worN=f, fs=FS)
    db = 20*np.log10(np.maximum(abs(h), 1e-300))
    pb, sb = db[f <= FP], db[f >= FST]
    radius = float(np.max(abs(zpk[1])))
    finite = bool(np.all(np.isfinite(h)) and np.all(np.isfinite(sos)))
    checks = {
        'sampling_rate_hz': FS,
        'filter_order': len(zpk[1]),
        'passband_min_db': float(pb.min()),
        'passband_max_db': float(pb.max()),
        'passband_ripple_db': float(pb.max()-pb.min()),
        'stopband_attenuation_db': float(-sb.max()),
        'maximum_pole_radius': radius,
        'stable': bool(radius < 1),
        'all_specs_pass': bool(finite and len(zpk[1]) >= 20 and radius < 1
                               and pb.min() >= -RP-1e-6
                               and pb.max() <= 1e-6 and -sb.max() >= AS-1e-6)
    }
    return checks, sos


# 3. Optimize the analog prototype, then apply the actual bilinear transform.
def design_filter():
    validate_settings()
    wp = 2*FS*np.tan(np.pi*FP/FS)
    ws = 2*FS*np.tan(np.pi*FST/FS) / wp
    # Extra samples near the passband edge resolve narrow high-order features.
    pass_f = np.unique(np.r_[np.linspace(0, FP, 1000),
                            np.linspace(max(0, FP-2000), FP, 1000)])
    stop_f = np.linspace(FST, FS/2-0.01, 1000)
    f = np.r_[pass_f, stop_f]
    analog_w = 2*FS*np.tan(np.pi*f/FS) / wp
    dp, ds = 1-10**(-RP/20), 10**(-AS/20)
    desired = np.r_[np.full(len(pass_f), 1-dp/2), np.zeros(len(stop_f))]
    weights = np.r_[np.full(len(pass_f), 2/dp), np.full(len(stop_f), 1/ds)]

    def cost(theta, p):
        mag, jac = analog_magnitude_and_jacobian(theta, analog_w)
        error = weights*(mag-desired)
        peak = max(float(np.max(abs(error))), 1e-300)
        scaled = error/peak
        norm = max(float(np.mean(abs(scaled)**p)**(1/p)), 1e-300)
        value = peak*norm
        derivative = (weights*np.sign(scaled)*abs(scaled)**(p-1)
                      / (len(error)*norm**(p-1)))
        return value, jac.T @ derivative

    theta = initial_guess(ws)
    m = ORDER//2
    # Bounds prevent overflow and undamped resonances; zeros stay in stopband.
    bounds = ([(-10, 4)]*m + [(-10, 5)]*m
              + [(np.log(ws*1.00001), 8)]*m + [(-150, 50)])
    if bounds[2*m][0] >= 8:
        raise ValueError('Band edges exceed the supported parameter range.')
    history = []
    for p in P_STEPS:
        before = cost(theta, p)[0]
        result = opt.minimize(cost, theta, args=(p,), jac=True,
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1200, 'maxls': 40,
                                       'maxcor': 20, 'ftol': 1e-11, 'gtol': 1e-7})
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError('Optimization produced invalid numbers.')
        if result.fun > before*(1+1e-8):
            raise RuntimeError('Optimization worsened its starting objective.')
        theta = result.x
        history.append({'p': p, 'theta': theta.copy(),
                        'objective': float(result.fun), 'iterations': int(result.nit),
                        'success': bool(result.success), 'message': str(result.message)})
        print(f'p={p:3d}: objective={result.fun:.6g}, '
              f'iterations={result.nit}, converged={result.success}', flush=True)
        if not result.success:
            print('  Warning:', result.message, '(final specifications still checked)')

    zn, pn, kn = unpack(theta)
    analog = (zn*wp, pn*wp, kn*wp**(len(pn)-len(zn)))
    zd, pd, kd = sig.bilinear_zpk(*analog, fs=FS)
    _, hp = sig.freqz_zpk(zd, pd, kd, worN=np.linspace(0, 2*np.pi*FP/FS, 65537))
    # A single DESIGN gain adjustment, not independent audio normalization.
    peak = float(np.max(abs(hp)))
    analog = (analog[0], analog[1], analog[2]/peak)
    digital = (zd, pd, kd/peak)
    checks, sos = check_specs(digital)
    if not checks['all_specs_pass']:
        raise RuntimeError('Do not report a successful design: '+json.dumps(checks))
    # Check that the prewarped analog response agrees with the digital response.
    test_f = np.linspace(0, FS/2-0.1, 10001)
    _, ha = sig.freqs_zpk(*analog, worN=2*FS*np.tan(np.pi*test_f/FS))
    _, hd = sig.sosfreqz(sos, worN=test_f, fs=FS)
    mapping_error = float(np.max(abs(ha-hd)))
    if mapping_error > 1e-7:
        raise RuntimeError('Analog/digital mapping consistency check failed.')
    checks['analog_digital_mapping_max_abs_error'] = mapping_error
    return {'analog': analog, 'digital': digital, 'sos': sos,
            'history': history, 'wp': wp, 'checks': checks}


# 4. Read and resample real samples; never just relabel a WAV header.
def load_audio(path):
    original_fs, raw = wavfile.read(path)
    original_dtype = raw.dtype
    if np.issubdtype(original_dtype, np.unsignedinteger):
        midpoint = (np.iinfo(original_dtype).max+1)/2
        x = (raw.astype(float)-midpoint)/midpoint
    elif np.issubdtype(original_dtype, np.signedinteger):
        x = raw.astype(float)/float(-np.iinfo(original_dtype).min)
    elif np.issubdtype(original_dtype, np.floating):
        x = raw.astype(float)
    else:
        raise ValueError('Unsupported WAV sample format.')
    channels = 1 if x.ndim == 1 else x.shape[1]
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError('Audio must be a nonempty finite mono/stereo recording.')
    if np.max(abs(x)) < 1e-12:
        raise ValueError('The recording is silent (or stereo channels cancel).')
    if original_fs != FS:
        factor = gcd(int(original_fs), FS)
        x = sig.resample_poly(x, FS//factor, int(original_fs)//factor)
        print(f'Resampled {original_fs} Hz to {FS} Hz; duration is preserved.')
    x = x * (SOURCE_PEAK/np.max(abs(x)))  # headroom before adding test noise
    return x, {'source_file': Path(path).name, 'original_sampling_rate_hz': int(original_fs),
               'original_channels': channels, 'duration_seconds': len(x)/FS}


def prepare_audio(source, add_noise=True):
    t = np.arange(len(source))/FS
    noise = NOISE_PEAK*np.sin(2*np.pi*NOISE_HZ*t) if add_noise else np.zeros_like(source)
    return source+noise, noise


def measure_audio(source, noise, noisy, filtered, design, add_noise):
    sos = design['sos']
    source_out, noise_out = sig.sosfilt(sos, source), sig.sosfilt(sos, noise)
    if np.max(abs(filtered-(source_out+noise_out))) > 1e-9:
        raise RuntimeError('Separate-component linearity check failed.')
    # Discard enough startup samples for roughly 100 dB pole decay.
    radius = design['checks']['maximum_pole_radius']
    settling = int(np.ceil(np.log(1e-5)/np.log(radius))) if radius > 0 else 0
    trim = max(int(.1*FS), settling)
    if trim >= len(source)//2:
        raise ValueError('Recording too short to measure after startup transient.')
    rms = lambda x: float(np.sqrt(np.mean(x[trim:]**2)))
    metrics = {'measurement_start_seconds': trim/FS,
               'source_component_rms_change_db': float(20*np.log10(rms(source_out)/rms(source))),
               'added_noise_hz': NOISE_HZ if add_noise else None,
               'added_noise_attenuation_db': None,
               'predicted_gain_at_noise_db': None,
               'full_speech_snr_db': None}
    if add_noise:
        metrics['added_noise_attenuation_db'] = float(20*np.log10(rms(noise)/rms(noise_out)))
        _, h = sig.sosfreqz(sos, worN=np.array([NOISE_HZ]), fs=FS)
        metrics['predicted_gain_at_noise_db'] = float(20*np.log10(abs(h[0])))
    # No clean reference was provided, so do not invent a speech SNR.
    return metrics


# 5. Figures for report/presentation. All comparisons use a common scale.
def make_plots(design, source, noisy, filtered, outdir):
    outdir = Path(outdir)
    sos, checks = design['sos'], design['checks']
    paths = []
    def save(fig, name):
        fig.tight_layout()
        path = outdir/name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    f = np.linspace(0, FS/2, 65537)
    _, h = sig.sosfreqz(sos, worN=f, fs=FS)
    db = 20*np.log10(np.maximum(abs(h), 1e-12))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a in axes:
        a.plot(f/1000, db, color='navy')
        a.axvspan(0, FP/1000, color='green', alpha=.07)
        a.axvspan(FP/1000, FST/1000, color='orange', alpha=.16)
        a.axvspan(FST/1000, FS/2000, color='red', alpha=.07)
        a.axhline(-RP, ls='--', color='green', label='Passband limit: -1 dB')
        a.axhline(-AS, ls='--', color='red', label='Stopband limit: -17 dB')
        a.axvline(FP/1000, color='gray', ls=':')
        a.axvline(FST/1000, color='gray', ls=':')
        a.set(xlabel='Frequency (kHz)', ylabel='Gain (dB)', ylim=(-110, 3))
        a.grid(alpha=.25)
    axes[0].set(xlim=(0, 24), title=f'{ORDER}th-order IIR response, {FS} Hz sampling')
    axes[1].set(xlim=(21, 24), title='Transition and stopband detail')
    axes[0].legend(fontsize=8, loc='lower left')
    save(fig, '01_filter_response.png')

    fig, a = plt.subplots(figsize=(9, 4))
    f = np.linspace(0, FS/2-.01, 6000)
    w = 2*FS*np.tan(np.pi*f/FS)/design['wp']
    for stage in design['history']:
        if stage['p'] in (2, 8, 32, 128):
            m = analog_magnitude_and_jacobian(stage['theta'], w)[0]
            a.plot(f/1000, 20*np.log10(np.maximum(m, 1e-12)), label=f"p = {stage['p']}")
    a.set(xlabel='Frequency (kHz)', ylabel='Magnitude (dB)', xlim=(21, 24), ylim=(-110, 3),
          title='Optimization stages (before final passband gain normalization)')
    a.axvline(FP/1000, color='gray', ls=':'); a.axvline(FST/1000, color='gray', ls=':')
    a.legend(); a.grid(alpha=.25)
    save(fig, '02_optimization_stages.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for a, zpk, title in zip(axes, [design['analog'], design['digital']],
                            ['Analog poles/zeros (s-plane)', 'Digital poles/zeros (z-plane)']):
        z, p, _ = zpk
        a.scatter(z.real, z.imag, facecolors='none', edgecolors='blue', label='Zeros')
        a.scatter(p.real, p.imag, marker='x', color='red', label='Poles')
        a.axvline(0, color='gray', lw=.7); a.axhline(0, color='gray', lw=.7)
        a.set(title=title, xlabel='Real part', ylabel='Imaginary part')
        a.legend(); a.grid(alpha=.25)
    circle = np.exp(1j*np.linspace(0, 2*np.pi, 500))
    axes[1].plot(circle.real, circle.imag, color='black', lw=.7)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_title(f"Digital poles: maximum radius {checks['maximum_pole_radius']:.5f}")
    save(fig, '03_pole_zero.png')

    t = np.arange(len(noisy))/FS
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    limit = 1.05*max(np.max(abs(noisy)), np.max(abs(filtered)))
    for a, y, title in zip(axes, [noisy, filtered], ['Noisy input', 'Filtered output']):
        a.plot(t, y, lw=.35)
        a.set(title=title, ylabel='Amplitude', ylim=(-limit, limit)); a.grid(alpha=.25)
    axes[1].set_xlabel('Time (seconds)')
    save(fig, '04_waveforms.png')

    # Same window, length, units, and scale for both amplitude spectra.
    window = sig.windows.hann(len(noisy), sym=False)
    f = np.fft.rfftfreq(len(noisy), 1/FS)
    spectra = []
    for y in (noisy, filtered):
        amp = 2*abs(np.fft.rfft(y*window))/window.sum()
        amp[0] /= 2
        if len(y) % 2 == 0:
            amp[-1] /= 2
        spectra.append(20*np.log10(np.maximum(amp, 1e-8)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a in axes:
        a.plot(f/1000, spectra[0], lw=.6, alpha=.8, label='Noisy input')
        a.plot(f/1000, spectra[1], lw=.6, alpha=.8, label='Filtered output')
        a.set(xlabel='Frequency (kHz)', ylabel='Amplitude (dBFS, peak reference)', ylim=(-140, 0))
        a.legend(); a.grid(alpha=.25)
    axes[0].set(xlim=(0, 24), title='Input/output DFT amplitude spectra')
    axes[1].set(xlim=(22, 24), title='High-frequency detail')
    save(fig, '05_dft_comparison.png')
    return paths


# 6. Save an auditable run: WAVs, coefficients, measurements, and figures.
def run_project(input_path=DEFAULT_WAV, output_dir='results', add_noise=True):
    print('Band-edge assumption: 22 kHz passband, 23 kHz stopband.', flush=True)
    print('Test mode:', 'recording + 23.05 kHz tone' if add_noise else 'recording only', flush=True)
    source, metadata = load_audio(input_path)
    design = design_filter()
    noisy, noise = prepare_audio(source, add_noise)
    filtered = sig.sosfilt(design['sos'], noisy)  # causal filtering, no filtfilt
    if not np.all(np.isfinite(filtered)):
        raise RuntimeError('Filtered audio contains invalid numbers.')
    # If needed, attenuate EVERY component equally to prevent WAV clipping.
    common_gain = min(1.0, .95/max(np.max(abs(noisy)), np.max(abs(filtered))))
    source, noise, noisy, filtered = [v*common_gain for v in (source, noise, noisy, filtered)]
    metrics = measure_audio(source, noise, noisy, filtered, design, add_noise)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, audio in [('source_resampled_48k.wav', source), ('noisy_input_48k.wav', noisy),
                        ('filtered_output_48k.wav', filtered)]:
        wavfile.write(outdir/name, FS, np.rint(audio*32767).astype(np.int16))
    # Also measure the tone in the actual quantized WAV files using the SAME
    # window and frequency. This is distinct from the isolated float component.
    if add_noise:
        start = int(round(metrics['measurement_start_seconds']*FS))
        def saved_tone_amplitude(path):
            rate, raw = wavfile.read(path)
            data = raw[start:].astype(float)/32768
            window = sig.windows.hann(len(data), sym=False)
            phase = np.exp(-2j*np.pi*NOISE_HZ*np.arange(len(data))/rate)
            return float(2*abs(np.sum(window*data*phase))/window.sum())
        ain = saved_tone_amplitude(outdir/'noisy_input_48k.wav')
        aout = saved_tone_amplitude(outdir/'filtered_output_48k.wav')
        metrics['saved_wav_tone_attenuation_db'] = float(20*np.log10(ain/max(aout, 1e-300)))
    np.save(outdir/'filter_sos.npy', design['sos'])
    np.savetxt(outdir/'filter_sos.csv', design['sos'], delimiter=',',
               header='b0,b1,b2,a0,a1,a2', comments='')
    report = {
        'assumptions': ['22 kHz interpreted as passband edge, not specified -3 dB point.',
                        'Stopband edge chosen as 23 kHz; confirm with instructor if needed.',
                        'Source may already be noisy; no clean-speech SNR is claimed.',
                        'Isolated component metrics use floats; saved-WAV tone attenuation uses PCM16 files.',
                        'Dense-grid verification is numerical, not a continuous-frequency proof.',
                        'Least p-norm optimization is local; global optimality is not claimed.'],
        'settings': {'fs': FS, 'fp': FP, 'fst': FST, 'rp': RP, 'as': AS, 'order': ORDER},
        'audio': metadata, 'added_test_noise': bool(add_noise),
        'common_audio_gain': common_gain,
        'filter_checks': design['checks'], 'audio_metrics': metrics,
        'optimization': [{k: v for k, v in row.items() if k != 'theta'} for row in design['history']]}
    (outdir/'measurements.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = [
        ('Sampling rate', FS, '48000 Hz', FS == 48000),
        ('Order', ORDER, '>=20', ORDER >= 20),
        ('Passband ripple (dB)', design['checks']['passband_ripple_db'], '<=1 dB', design['checks']['passband_ripple_db'] <= RP),
        ('Stopband attenuation (dB)', design['checks']['stopband_attenuation_db'], '>=17 dB', design['checks']['stopband_attenuation_db'] >= AS),
        ('Maximum pole radius', design['checks']['maximum_pole_radius'], '<1', design['checks']['stable'])]
    with (outdir/'requirements_vs_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Measurement', 'Result', 'Requirement', 'Pass'])
        writer.writerows(rows)
    plots = make_plots(design, source, noisy, filtered, outdir)
    print('\nVERIFIED FILTER RESULTS')
    for label, value, requirement, passed in rows:
        print(f'{label}: {value:.6g} | {requirement} | {"PASS" if passed else "FAIL"}')
    if add_noise:
        print(f"Isolated {NOISE_HZ:g} Hz tone attenuation (float): {metrics['added_noise_attenuation_db']:.2f} dB")
        print(f"Tone attenuation measured in saved WAVs: {metrics['saved_wav_tone_attenuation_db']:.2f} dB")
    print('No separate output-volume normalization was applied.')
    print('Saved results in:', outdir.resolve())
    return {'report': report, 'design': design, 'plots': plots, 'output_dir': outdir}


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wav', nargs='?', default=str(folder/DEFAULT_WAV),
                        help='Source WAV; defaults to team10_noisy_48k.wav beside this script.')
    parser.add_argument('--output-dir', default=str(folder/'results'),
                        help='Folder for WAVs, graphs, coefficients, and measurements.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--add-noise', dest='add_noise', action='store_true',
                      help='Add the known 23.05 kHz demonstration tone (default).')
    mode.add_argument('--no-add-noise', dest='add_noise', action='store_false',
                      help='Filter the supplied recording without adding a test tone.')
    parser.set_defaults(add_noise=True)
    args = parser.parse_args()
    run_project(args.wav, args.output_dir, args.add_noise)
