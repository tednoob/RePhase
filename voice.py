from pydub import AudioSegment
from glob import iglob
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
from scipy import ndimage
from scipy import signal
from scipy import interpolate
from scipy import fft

# https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html
# https://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
# https://en.wikipedia.org/wiki/Spectral_density
# https://en.wikipedia.org/wiki/Formant

def flatten(segments):
    output = []
    for segment in segments:
        output.extend(segment.get_array_of_samples())
    return np.array(output, dtype=np.int16)


def to_segment(samples, frame_rate=48000, sample_width=2, channels=1):
    """
    data: raw audio data (bytes)
    sample_width: 2 byte (16 bit) samples
    frame_rate: samples per second
    channels: Number of channels

    """
    data = samples.tobytes(order='C')
    print(len(data))
    print(frame_rate, sample_width, channels)
    return AudioSegment(data=data, sample_width=sample_width, frame_rate=frame_rate, channels=channels)


def amplify(data, window_size=24000):
    a = np.abs(data)
    data_len = len(data)
    output = np.zeros_like(a)
    for i in range(data_len):
        window_start = max(i-window_size//2, 0)
        window_end = min(i+window_size//2, data_len-1)
        tmp = a[window_start:window_end]
        tmp_max = np.max(tmp)
        if tmp_max > 0:
            output[i] = 0.75*data[i]*np.iinfo(np.int16).max / tmp_max
    return output


def transform(segments, opt_log10=False, opt_abs=True, opt_gaussian=None, nperseg=4096, noverlap=4000):
    x = flatten(segments)
    fs = segments.frame_rate
    x = amplify(x, window_size=fs//2)
    t = np.linspace(0, len(x)/segments.frame_rate, len(x))

    # All of the audio clip
    f1, t1, Zxx = scipy.signal.stft(x, fs, nperseg=nperseg, noverlap=noverlap)
    if opt_abs:
        Zxx = np.abs(Zxx)
    if opt_log10:
        Zxx = np.log10(Zxx)
    if opt_gaussian is not None:
        Zxx = scipy.ndimage.gaussian_filter(Zxx, opt_gaussian)

    return t, f1, t1, Zxx


def amplify_segments(segments):
    x = flatten(segments)
    x = amplify(x, window_size=24000)
    output = to_segment(x, segments.frame_rate, segments.sample_width, segments.channels)
    file_handle = output.export("D:\\output.mp3", format="mp3")


def draw_fft(segments, x=None):
    if x is None:
        x = flatten(segments)
    N = len(x)
    #s = np.abs(scipy.fft.fft(x))[:N//2]

    T = N/segments.frame_rate
    print(N, T, N*T)
    t = np.linspace(0.0, T, N)
    sp = scipy.fft.fft(x)
    print(dir(sp))
    print(t.shape[-1], segments.frame_rate)
    freq = scipy.fft.fftfreq(t.shape[-1], d=1/segments.frame_rate)
    plt.plot(freq[:N//2], np.abs(sp.real[:N//2]))
    plt.show()


def spectrogram(segments, opt_log10=False, opt_show=True):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    
    """
    x = flatten(segments)
    fs = segments.frame_rate
    print(len(x)/fs, len(x), fs)
    t, f1, t1, Zxx = transform(segments, opt_log10=opt_log10, opt_abs=True)
    
    frequency_precision = f1[1]-f1[0]
    print("".join(str(x) for x in ["Frequency precision: ", frequency_precision, "Hz"]))

    plt.pcolormesh(t1, f1, Zxx, shading='flat')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if opt_show:
        plt.show()


def play(segments):
    from pydub import AudioSegment
    from pydub.playback import play
    play(segments)


class Interpolater:

    def __init__(self, segments):
        self.fs = segments.frame_rate
        self.data = flatten(segments)
        self.freq, self.times, Zxx = scipy.signal.stft(self.data, fs=self.fs, nperseg=4096)
        self.Zxx = np.abs(Zxx)
        self.fun = scipy.interpolate.interp2d(self.times, self.freq, self.Zxx, kind='cubic')
        print(self.freq.shape, self.times.shape)

    def _index(self, value, vector):
        low_index = 0
        high_index = 0
        max_index = len(vector)-1
        for v in vector:
            high_index+=1
            if v >= value:
                break
            low_index+=1
        if low_index > max_index:
            low_index = max_index
        if high_index > max_index:
            high_index = max_index
        return (low_index, high_index), (vector[low_index], vector[high_index])

    def intensity(self, time, freq):
        return self.fun(time, freq)

    def get_signal(self, freq, times):
        amplitude = np.zeros_like(times)
        sig = np.real(np.exp( (times * freq * 2 * np.pi * 1j) + np.random.random() * 2 * np.pi * 1j))
        for ix, a in enumerate(amplitude):
            amplitude[ix] = self.intensity(times[ix], freq) * sig[ix]
        return amplitude


def static_tones(segments):

    interp = Interpolater(segments)
    max_t = len(interp.data)/interp.fs
    all_times = np.transpose(np.linspace(0, max_t, len(interp.data)))
    
    a = np.zeros_like(all_times)
    f = 50
    while f < 2000:
    #for f in [123, 232, 344, 373, 495, 719, 827, 1037, 1260]:
        print(f)
        a += interp.get_signal(f, all_times)
        f += 30+100*np.random.random()
    a = np.iinfo(np.int16).max * 0.75 * a / np.max(np.abs(a))
    a = a.astype("int16")

    output = to_segment(a, segments.frame_rate, segments.sample_width, segments.channels)
    spectrogram(output)

    file_handle = output.export("D:\\output.mp3", format="mp3")

    plt.plot(all_times, a)
    plt.show()


def remove_phase(segments):
    fs=segments.frame_rate
    data = flatten(segments)
    freq, times, Zxx = scipy.signal.stft(data, fs, nperseg=1024)
    t, x = scipy.signal.istft(np.abs(Zxx), fs)
    x = x.astype("int16")
    output = to_segment(x, segments.frame_rate, segments.sample_width, segments.channels)
    file_handle = output.export("D:\\output.mp3", format="mp3")


def save(segments):
    print(segments.sample_width, segments.frame_rate, segments.channels)
    data = flatten(segments)
    fs = segments.frame_rate
    _, _, Zxx = scipy.signal.stft(data, fs)
    t, x = scipy.signal.istft(Zxx, fs)
    x = x.astype("int16")
    
    output = to_segment(x, segments.frame_rate, segments.sample_width, segments.channels)
    file_handle = output.export("D:\\output.mp3", format="mp3")


def smooth(vec, window_size=5):
    ret = np.zeros_like(vec)
    for i, _ in enumerate(vec):
        tmp = vec[i-window_size//2:i+window_size//2]
        ret[i] = sum(tmp)/max(1, len(tmp))
    return ret


def find_max_ix(vec, window_size=5):
    ret = []
    for i, _ in enumerate(vec):
        tmp = vec[i-window_size//2:i+window_size//2]
        if tmp.size > 0 and vec[i] == max(vec[i-window_size//2:i+window_size//2]):
            ret.append(i)
    return ret


def find_max_smooth(x_values, y_values, window_size=5):
    vec = smooth(y_values, window_size=window_size)
    y_max = np.max(y_values)
    max_xs = []
    max_ys = []
    for ix in find_max_ix(vec, window_size=5):
        if y_values[ix] > 0.01*y_max:
            max_xs.append(x_values[ix])
            max_ys.append(y_values[ix])
    return max_xs, max_ys


def find_max_interp(x_values, y_values, cutoff=None, min_x=20, max_x=2000, resolution=1):
    y_fun = Spline(x_values, y_values)

    x = np.linspace(min_x, max_x, int((max_x - min_x)/resolution))
    y = np.array(list(y_fun(f) for f in x))
    y_prime = np.array(list(y_fun.derivate(f) for f in x))

    max_xs = []
    max_ys = []
    for i in range(1, len(y_prime)):
        if np.sign(y_prime[i]) < np.sign(y_prime[i-1]):
            x_tmp = x[i-1] + (x[i]-x[i-1])/2
            y_tmp = y_fun(x_tmp)
            if cutoff is None or y_tmp >= cutoff:
                max_xs.append(x_tmp)
                max_ys.append(y_tmp)

    return max_xs, max_ys


class Spline:

    def __init__(self, x, y, s=480):
        self.x = x
        self.y = y
        self.spl = scipy.interpolate.splrep(self.x, self.y, s=s)
        self.der = scipy.interpolate.splder(self.spl, 1)

    def __call__(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return 0
        return scipy.interpolate.splev(x, self.spl)

    def derivate(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return 0
        return scipy.interpolate.splev(x, self.der)


class Particle:

    def __init__(self, time, frequency, amplitude, smooth_amplitude=None):
        self.time = time
        self.frequency = frequency
        self.amplitude = amplitude
        self.smooth_amplitude = smooth_amplitude if smooth_amplitude is not None else amplitude
        self.prev=None
        self.next=None

    def delta_freq(self, particle):
        return np.abs(particle.frequency-self.frequency)

    def __len__(self):
        l = 0
        step = self
        while step is not None:
            step = step.next
            l += 1
        return l

    @property
    def duration(self):
        time, _, _ = self.line()
        if time:
            return time[-1]-time[0]
        return 0

    def line(self):
        times = []
        frequencies = []
        amplitudes = []
        step = self
        while step is not None:
            times.append(step.time)
            frequencies.append(step.frequency)
            amplitudes.append(step.smooth_amplitude)
            step = step.next
        return times, frequencies, amplitudes

    def interpolate(self, times):
        try:
            amplitude = np.zeros_like(times)
            line_times, line_freqs, line_amps = self.line()
            #freq = scipy.interpolate.interp1d(line_times, line_freqs, kind='cubic', fill_value="extrapolate")
            #amp = scipy.interpolate.interp1d(line_times, line_amps, kind='cubic', fill_value="extrapolate")
            freq = Spline(line_times, line_freqs)
            amp = Spline(line_times, line_amps)
            cum_arg = 0
            first = True
            cum_t = 0
            for i, t in enumerate(times):
                delta_t = t-cum_t
                if t > line_times[0] and t < line_times[-1]:
                    f = freq(t)
                    a = amp(t)
                    if first:
                        cum_arg = t*f
                        first = False
                    else:
                        cum_arg += f * delta_t
                    amplitude[i] = np.real(a*np.exp(cum_arg * 2 * np.pi * 1j))
                cum_t += delta_t
            return amplitude
        except:
            return np.zeros_like(times)

    def plot(self, times):
        try:
            plot_times = []
            amplitude = []
            frequency = []
            line_times, line_freqs, line_amps = self.line()
            #freq = scipy.interpolate.interp1d(line_times, line_freqs, kind='cubic', fill_value="extrapolate")
            #amp = scipy.interpolate.interp1d(line_times, line_amps, kind='cubic', fill_value="extrapolate")
            freq = Spline(line_times, line_freqs)
            amp = Spline(line_times, line_amps)
            for i, t in enumerate(times):
                if t > line_times[0] and t < line_times[-1]:
                    plot_times.append(t)
                    frequency.append(freq(t))
                    amplitude.append(amp(t))

            colors = "bgrcmykw"
            color = random.choice(colors)
            plt.plot(plot_times, frequency, '-' + color)
            plt.plot(line_times, line_freqs, 'o' + color)
        except:
            pass

def filter_particles(particles, duration=0.1, minlen=5):
    all_particles = []
    for particle in particles:
        if particle.duration >= duration and len(particle) >= minlen:
            all_particles.append(particle)
    return all_particles


def find_closest_index(vec, value):
    tmp = np.abs(vec - value)
    return tmp.argmin()


def find_base_frequency(max_xs, max_ys, tolerance=0.01, guess=None):

    # This is shady and unreliable.
    base_freq = 1
    if len(max_ys) == 0:
        return base_freq
    
    max_y = max(max_ys)
    for i, _ in enumerate(max_xs):
        if max_ys[i] >= tolerance*max(max_ys):
            base_freq = max_xs[i]
            break
    return base_freq


def find_base_binning(max_xs, max_ys, resolution=5, low_base=75, high_base=200, min_x=20, max_x=2000):
    points = list()
    for ix, x in enumerate(max_xs):
        if x >= min_x and x < max_x:
            low_range = max(int(np.floor(x/high_base)), 1)
            high_range = int(np.ceil(x/low_base))
            for n in range(low_range, high_range):
                points.append((x/n, max_ys[ix]/n, max_ys[ix], n))
    points.sort(key=lambda x: x[0])

    guesses = list()
    low_ix = 0
    high_ix = 0
    max_ix = len(points)-1
    f = low_base
    while f < high_base:
        while low_ix < max_ix and points[low_ix][0] < f-resolution/2:
            low_ix += 1
        while high_ix < max_ix and points[high_ix][0] < f+resolution/2:
            high_ix += 1
        window = points[low_ix:high_ix]
        win_len = len(window)
        if win_len > 0:
            freq = sum(x[0] for x in window)/win_len
            amplitude = sum(x[1] for x in window)
            guesses.append((freq, amplitude))
        f += resolution

    guesses.sort(key=lambda x: x[1], reverse=True)
    if len(guesses):
        return guesses[0][0]
    return find_base_frequency(max_xs, max_ys)


def find_peaks(x_values, y_values, min_x=20, max_x=2000, cutoff=None, resolution=0.1, harmonics=True, tolerance=0.1, base_guess=None):
    if min_x is None:
        min_x = x_values[0]

    if max_x is None:
        max_x = x_values[-1]


    max_xs, max_ys = find_max_interp(x_values, y_values, cutoff=cutoff, min_x=min_x, max_x=max_x, resolution=resolution)
    #max_xs, max_ys = find_max_smooth(x_values, y_values)

    if not harmonics or len(max_xs) == 0:
        return max_xs, max_ys

    harm_xs = []
    harm_ys = []
    #base_freq = find_base_frequency(max_xs, max_ys, tolerance=tolerance, guess=None)
    base_freq = find_base_binning(max_xs, max_ys)


    low_ix = 0
    high_ix = 0
    max_ix = len(max_xs)-1

    for n in range(1, int(max_x/base_freq)):
        freq = n*base_freq
        amplitude = max_ys[find_closest_index(max_xs, freq)]

        while low_ix < max_ix and max_xs[low_ix] < freq-20:
            low_ix += 1
        while high_ix < max_ix and max_xs[high_ix] < freq+20:
            high_ix += 1
        window = max_ys[low_ix:high_ix]
        win_len = len(window)
        if win_len > 0:
            amplitude = max(x for x in window)

        harm_xs.append(freq)
        harm_ys.append(amplitude)
    
    return harm_xs, harm_ys


def find_particles(f1, t1, Zxx, resolution=0.1, step_size=5, amplitude_limit=0.05, match_freq=50):
    """
    amplitude_limit = 0.3 is really cool with log10.
    opt_gaussian defaults to None but you can provide [t, f] to apply window size e.g [0, 100]

    """
    
    times = []
    frequencies = []
    particles = []
    prev_particles = []
    amplitude_criteria = amplitude_limit * np.max(Zxx)
    guess = None
    for i in range(0, len(t1), step_size):
        print(i, len(t1))
        time = t1[i]
        new_particles = []

        # should be in a window
        max_z = np.max(Zxx)
        max_fs, max_amps = find_peaks(f1, Zxx[:, i], max_x=3000, cutoff=None, resolution=resolution, harmonics=False, base_guess=guess)
        for ix, frequency in enumerate(max_fs):
            amplitude = max_amps[ix]
            if max_amps[ix] > amplitude_criteria:
                times.append(time)
                frequencies.append(frequency)
                new_particles.append(Particle(time=time, frequency=frequency, amplitude=amplitude))
        if len(max_fs) > 0:
            guess = max_fs[0]

        # Todo: improve complexity in matching
        new_particles.sort(key=lambda x: x.smooth_amplitude, reverse=True)
        for part in new_particles:
            tmp_prev = [p for p in prev_particles if p.next == None]
            if tmp_prev:
                min_prev = tmp_prev[0]
                min_delta = tmp_prev[0].delta_freq(part)
                for pix in range(1, len(tmp_prev)):
                    tmp_delta = tmp_prev[pix].delta_freq(part)
                    if tmp_prev[pix].delta_freq(part) < min_delta:
                        min_prev = tmp_prev[pix]
                        min_delta = tmp_delta
                if min_delta > match_freq: #Hz
                    continue
                min_prev.next = part
                part.prev = min_prev
                if min_prev.prev is None:
                    particles.append(min_prev)
        prev_particles = new_particles

    #plt.plot(times, frequencies, 'ro')

    return particles


def build_signal(t, particles):
    all_signals = np.zeros_like(t)
    print(t[0], t[-1])
    for ix, particle in enumerate(particles):
        print(ix, len(particles))
        all_signals += particle.interpolate(t)

    all_signals = np.iinfo(np.int16).max * all_signals / np.max(np.abs(all_signals))
    all_signals = all_signals.astype("int16")
    return amplify(all_signals)


def rebuild_signal(segments, amplitude_limit=0.01, step_size=1, resolution=0.01):
    spectrogram(segments)
    t, f1, t1, Zxx = transform(segments)
    particles = find_particles(f1, t1, Zxx, amplitude_limit=amplitude_limit, step_size=step_size, resolution=resolution)
    particles = filter_particles(particles, duration=0, minlen=4)
    all_signals = build_signal(t, particles)

    output = to_segment(all_signals, segments.frame_rate, segments.sample_width, segments.channels)
    #file_handle = output.export("D:\\output.mp3", format="mp3")
    print("output written")

    plt.show()

    spectrogram(output, opt_show=False)
    for ix, particle in enumerate(particles):
        print(ix, len(particles))
        particle.plot(t)
    plt.show()


def decode_image_from_stackexchange():
    # Image from https://dsp.stackexchange.com/questions/47304/stft-computation

    from PIL import Image
    import PIL.ImageOps
    Zxx = np.array(PIL.ImageOps.flip(PIL.ImageOps.invert(Image.open("D:\\so_sample.png").convert('L'))))
    f1 = np.linspace(0, 4000, Zxx.shape[0])
    t1 = np.linspace(0, 3, Zxx.shape[1])
    t = np.linspace(0, 3, 3*16000)
    print(Zxx.shape, f1.shape, t1.shape)

    particles = find_particles(f1, t1, Zxx, amplitude_limit=0, step_size=1)
    #particles = filter_particles(particles, 0.01)

    all_signals = build_signal(t, particles)

    output = to_segment(all_signals, 16000, 2, 1)
    file_handle = output.export("D:\\output.mp3", format="mp3")
    print("output written")

    #spectrogram(output, opt_show=False)
    for ix, particle in enumerate(particles):
        print(ix, len(particles))
        particle.plot(t)
    plt.show()


def generate_particle(t_start, t_end, freq_fun, amplitude):
    t = t_start
    frequency = freq_fun(t)
    particle = Particle(t, frequency, amplitude)
    current = particle
    while t < t_end:
        t += 0.1
        freq = freq_fun(t)
        next_part = Particle(t, freq_fun(t), amplitude)
        current.next = next_part
        next_part.prev = current
        current = next_part

    return particle


def test_particle():
    t_start = 0
    t_end = 3
    sample_rate = 16000

    particles = list()
    base_freq = 124
    amplitude = [3, 10, 18, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for n in range(1, 20):
        particles.append(generate_particle(t_start, t_end, freq_fun=lambda x: (0.75 + 0.25*(x%1))*n*base_freq, amplitude=amplitude[n]))

    t = np.linspace(t_start, t_end, int((t_end-t_start)*sample_rate))
    all_signals = build_signal(t, particles)
    segments = to_segment(all_signals, sample_rate, 2, 1)
    rebuild_signal(segments, amplitude_limit=0, step_size=5, resolution=1)


enable_profiling = False
if enable_profiling:
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

#DATA_FILES_GLOB="D:\\output*"
DATA_FILES_GLOB="D:\\Desktop\\soundboard\\Dahlgren\\jajjemen.flac"
#DATA_FILES_GLOB="D:\\Desktop\\soundboard\\Alekanderu\\*skratt*.flac"
#DATA_FILES_GLOB="D:\\Desktop\\soundboard\\Misc\\*.flac"
#DATA_FILES_GLOB="E:\\monoton.flac"
for file in iglob(DATA_FILES_GLOB):
    print(file)
    segments = AudioSegment.from_file(file, file.split(".")[-1])
    #draw_fft(segments)
    #save(segments)
    #spectrogram(segments)
    #static_tones(segments)
    #remove_phase(segments)
    #rebuild_signal(segments, amplitude_limit=0, step_size=5, resolution=1)
    test_particle()
    #test_find_base_frequency(segments)
    #decode_image_from_stackexchange()
    #amplify_segments(segments)

if enable_profiling:
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
