spectrogram = spectrogram';
stft = spectrogram.*phase;
fs = 16000;
wlen = fs*0.025;
nfft = 1024;
re = istft2017(stft, wlen, nfft, fs);
sound(re, fs);