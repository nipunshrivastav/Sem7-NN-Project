%% Cross Spectral Features
%% The magnitude square coherence features for each pair of channels in the DEAP Dataset are extracted
%% No. of patterns = 32 persons * 40 videos = 1280 patterns
%% No. of features = 32 channels choose 2 = 496 features
%% The output of this script is 1280*486 array  
 
Fs = 128; %Sampling Frequency
hwin_size = 2*Fs; % 2 seconds
overlap_size = hwin_size/2;    % 50 percent 

msce_features_matrix = zeros(32*40,496);

for person = 1:32 %for each person
    varName = 'data_preprocessed_matlab\s';
    if person<10
        varName = [varName,'0',num2str(person),'.mat'];
    else
        varName = [varName,num2str(person),'.mat'];
    end
    load(varName); %load data
    data_siz = size(data);
    display(person);
    for i = 1:data_siz(1)   % For each video
        count = 1;
        for j = 1:32 % For each channel
            for k = j+1:32 % For each channel pair j,k
             
				signal1 = data(i,j,:);
                signal2 = data(i,k,:);
				
				% The magnitude squared coherence
                [Pxy,F] = mscohere(signal1,signal2,hamming(hwin_size),overlap_size,256,Fs,'onesided');
				p = bandpower(Pxy, F, 'psd');
				
				% Store the features
                msce_features_matrix(40*(person-1)+i,count) = p_theta;
                count = count+1;
            end
        end
    end
end