%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Inverse Short-Time Fourier Transform        %
%               with MATLAB Implementation             %
%                                                      %
% Author: Jeff Wu, Allen Chou       04/24/17           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y, t] = istft2017(stft, wlen, nfft, fs)

% function: [x, t] = istft(stft, h, nfft, fs)
% stft - STFT matrix (only unique points, time across columns, freq across rows)
% h - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
% x - signal in the time domain
% t - time vector, s

% estimate the length of the signal
num_frame = size(stft, 2);

% hopsize must be half of window to perfectly reconstruct
h = wlen/2;

% set length of output
y = zeros(h*(num_frame-1)+nfft,1);



% perform STFT
for mm = 1:num_frame
    t_start = (mm-1)*h;  
    X = stft(:,mm);
    Y = [X; conj(X( (end-1):-1:2))];
    y_win = ifft(Y); %ifft
    
    % overlap-add 
    tt2 = 1:nfft-1;
    y(t_start+tt2) = y(t_start+tt2)+y_win(1:end-1); %left


end

% calculate the time vector
actxlen = length(y);                % find actual length of the signal
t = (0:actxlen-1)/fs;               % generate time vector

end