"""Cross-platform (Windows/Linux/macOS) version of fetch_data.sh: downloads the 15
high-dimensional datasets of the scikit-feature repository into data/hd/."""
import os, urllib.request
NAMES = ['leukemia', 'colon', 'lung', 'lymphoma', 'GLIOMA', 'Carcinom', 'nci9', 'COIL20', 'ORL', 'Yale',
         'warpPIE10P', 'warpAR10P', 'USPS', 'Isolet', 'PCMAC']
URL = 'https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/{}.mat'
os.makedirs(os.path.join('data', 'hd'), exist_ok=True)
for n in NAMES:
    f = os.path.join('data', 'hd', f'{n}.mat')
    if not os.path.exists(f):
        print('downloading', n, flush=True)
        urllib.request.urlretrieve(URL.format(n), f)
    print(f, os.path.getsize(f), 'bytes')
