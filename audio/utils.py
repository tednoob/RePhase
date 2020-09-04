import numpy as np

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
