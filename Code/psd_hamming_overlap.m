%% Extract Power Spectral Density Features from DEAP Dataset
%% The spectral powers from each of delta, alpha, beta and gamma bands are calculated for each channel
%% pwelch function is used for PSD. A hamming window of size 2 seconds with 50% overlap is used and one sided periodogram is calculated.
%% Moreover virtual channels are constructed by using longitudinal and transverse bipolar montages
%% Number of patterns = 32 persons * 40 videos = 1280
%% Number of features = (32 channels + 61 virtual channels)*4 = 372
%% The output of this script is 1280*372 array

clear all;clc; close all;

%Constants
Fs = 128; %Sampling Frequency
hwin_size = 2*Fs; % 2 seconds
overlap_size = hwin_size/2;    % 50 percent 
% Channel indices for bipolar montage
diff_vect = [1,4,8,12,1,5,9,2,3,7,11,6,19,24,16,23,18,20,25,29,17,22,27,17,21,26,30,1,2,4,3,19,20,5,6,23,8,7,24,25,9,10,28,12,11,16,29,13,14,15,4,3,5,6,8,7,9,10,12,11,14];
diff_vect2 = [4,8,12,14,5,9,14,3,7,11,13,10,24,16,15,28,20,25,29,31,22,27,32,21,26,30,32,17,18,3,19,20,21,6,23,22,7,24,25,26,10,28,27,11,16,29,30,31,15,32,21,20,22,23,26,25,27,28,30,29,32];    

feature_vector = zeros(32*40,4*(32+length(diff_vect)));


for person = 1:32    % For each person
    varName = 'data_preprocessed_matlab\s'; 
	% Load data
    if person<10
        varName = [varName,'0',num2str(person),'.mat'];
    else
        varName = [varName,num2str(person),'.mat'];
    end
    load(varName);
    data_siz = size(data);
    display(person,'person');
	
    for i = 1:data_siz(1)   % For each video
        for j = 1:32 % For each channel
            signal = data(i,j,:);
			
			% Power Spectral Density
            [pxx,f] = pwelch(signal,hwin_size,overlap_size,256,Fs,'onesided'); % 256 is DFT bin size
            
			% Spectral powers
			p_theta = bandpower(pxx, f,[4, 8], 'psd');
            p_alpha = bandpower(pxx, f,[8, 13], 'psd');
            p_beta = bandpower(pxx, f,[13, 30], 'psd');
            p_gamma = bandpower(pxx, f,[36, 44], 'psd');
            
			% Store the features
			feature_vector(40*(person-1)+i,4*j-3) = p_theta;
            feature_vector(40*(person-1)+i,4*j-2) = p_alpha;
            feature_vector(40*(person-1)+i,4*j-1) = p_beta;
            feature_vector(40*(person-1)+i,4*j) = p_gamma;
        end
		
		% For each virtual channel
        for index=1:length(diff_vect)
            
			signal = data(i,diff_vect,:)-data(i,diff_vect2,:);
            
			% Power Spectral Density
			[pxx,f] = pwelch(signal,hwin_size,overlap_size,256,Fs,'onesided'); % 256 is DFT bin size
            
			% Spectral powers
			p_theta = bandpower(pxx, f,[4, 8], 'psd');
            p_alpha = bandpower(pxx, f,[8, 13], 'psd');
            p_beta = bandpower(pxx, f,[13, 30], 'psd');
            p_gamma = bandpower(pxx, f,[36, 44], 'psd');
            
			% Store the features
			feature_vector(40*(person-1)+i,128+4*index-3) = p_theta;
            feature_vector(40*(person-1)+i,128+4*index-2) = p_alpha;
            feature_vector(40*(person-1)+i,128+4*index-1) = p_beta;
            feature_vector(40*(person-1)+i,128+4*index) = p_gamma;
        end
    end
end