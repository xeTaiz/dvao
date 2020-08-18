from peak_finder import get_persistent_homology
vol
vals, bins = np.histogram(vol, 200)
len(vals)
peaks = get_persistent_homology(vals)
peaks
len(peaks)
peaks[0]
peaks[0].born
peaks[0].died
peaks[1].died
peaks[1].born
import matplotlib.pyplot as plt
plt.hist(vals)
plt.show()
plt.hist()
plt.hist(vals, 200)
plt.show()
peak_vals = list(map(lambda p: vals[p.born], peaks))
peak_vals
peak_bins = list(map(lambda p: bins[p.born], peaks))
peak_bins
list(map(lambda p: p.born, peaks))
list(map(lambda p: (p.born, p.get_persistence(vals), peaks))
)
list(map(lambda p: (p.born, p.get_persistence(vals)), peaks))
list(map(lambda p: (p.born, p.get_persistence(vals), (bins[p.born] + bins[p.born+1)/2.0), peaks))
list(map(lambda p: (p.born, p.get_persistence(vals), (bins[p.born] + bins[p.born+1])/2.0), peaks))
vol.min()
import readline
readline.write_history_file('extract_peaks.py')
