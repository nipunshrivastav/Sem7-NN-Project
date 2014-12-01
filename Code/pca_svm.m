%% Implements PCA followed by SVM
%% It is assumed that features are extracted and stored as msce_features_matrix variable

%% Visualize
numOfPersons =32;
input_data = msce_features_matrix(1:numOfPersons*40,:);  %Put the data matrix here
data = bsxfun(@minus, input_data, mean(input_data)); 
data = bsxfun(@times, data, 1./std(data)); 
tempo = size(input_data); 
m = tempo(1); % Number of patterns

%% Pre-Processing
Sigma = (data'*data) ./ m;
[U, S, V] = svd(Sigma);
%% Compressing data
k = 40; % Choose the number of dimensions in the output
U_red = U(:,1:k);
x_red = zeros(m,k);

for ni = 1:m
    x_red(ni,:) = (U_red'*data(ni,:)')'; 
end

% x_red is the input with 'k' PCA dimensions


%% SVM Train, Predict and Validate
tr_percent_array = [1]; % Percentage of training data to be included

mean_ac_v = zeros(1,length(tr_percent_array)); 
std_ac_v = zeros(1,length(tr_percent_array)); 
mean_ac_a = zeros(1,length(tr_percent_array)); 
std_ac_a = zeros(1,length(tr_percent_array));
mean_ac_d = zeros(1,length(tr_percent_array)); 
std_ac_d = zeros(1,length(tr_percent_array)); 
mean_ac_l = zeros(1,length(tr_percent_array)); 
std_ac_l = zeros(1,length(tr_percent_array)); 

for ac = 1: length(tr_percent_array)
    tr_percent = 0.7; %The ratio of training to total data
    ac_vect_v = zeros(1,10);
    ac_vect_a = zeros(1,10);
    ac_vect_d = zeros(1,10);
    ac_vect_l = zeros(1,10);
    
    for trial = 1:15
        % Randomly partition data into training and test 
        index_vector = randperm(numOfPersons*40); 
        % Create empty arrays of required size 
        training_data = zeros(numOfPersons*40*tr_percent,k);
        v_tr_l = zeros(numOfPersons*40*tr_percent,1);
        a_tr_l= zeros(numOfPersons*40*tr_percent,1);
        d_tr_l = zeros(numOfPersons*40*tr_percent,1);
        l_tr_l = zeros(numOfPersons*40*tr_percent,1);
        test_data = zeros(numOfPersons*40*(1-tr_percent),k);
        v_te_l = zeros(numOfPersons*40*(1-tr_percent),1);
        a_te_l= zeros(numOfPersons*40*(1-tr_percent),1);
        d_te_l = zeros(numOfPersons*40*(1-tr_percent),1);
        l_te_l = zeros(numOfPersons*40*(1-tr_percent),1);
        % Populate the training and test arrays
        for i=1:length(index_vector)
            if i<=tr_percent*length(index_vector)
                training_data(i,:) = x_red(index_vector(i),:);
                v_tr_l(i) = valence_labels(index_vector(i));
                a_tr_l(i) = arousal_labels(index_vector(i));
                d_tr_l(i) = dominance_labels(index_vector(i));
                l_tr_l(i) = liking_labels(index_vector(i));
            else
                test_data(i-tr_percent*length(index_vector),:) = x_red(index_vector(i),:);
                v_te_l(i-tr_percent*length(index_vector)) = valence_labels(index_vector(i));
                a_te_l(i-tr_percent*length(index_vector)) =arousal_labels(index_vector(i));
                d_te_l(i-tr_percent*length(index_vector)) =dominance_labels(index_vector(i));
                l_te_l(i-tr_percent*length(index_vector)) =liking_labels(index_vector(i));
            end
        end
        cur_len = floor(tr_percent_array(ac)*length(l_tr_l));
        % Train each of valence, arousal, dominance and liking separately
        v_modelGauss = svmtrain(double(v_tr_l(1:cur_len,:)),training_data(1:cur_len,:),'-c 11 -g .004');
        a_modelGauss = svmtrain(double(a_tr_l(1:cur_len,:)),training_data(1:cur_len,:), '-c 11 -g 0.004');
        d_modelGauss = svmtrain(double(d_tr_l(1:cur_len,:)),training_data(1:cur_len,:), '-c 11 -g 0.004');
        l_modelGauss = svmtrain(double(l_tr_l(1:cur_len,:)),training_data(1:cur_len,:), '-c 0.1 -g 0.0025');
        % Test the trained model and store accuracies
        [v_predict_label_gauss, v_accuracy_gauss, v_prob_values_gauss] = svmpredict(double(v_te_l), test_data, v_modelGauss);
        [a_predict_label_gauss, a_accuracy_gauss, a_prob_values_gauss] = svmpredict(double(a_te_l), test_data, a_modelGauss);
        [d_predict_label_gauss, d_accuracy_gauss, d_prob_values_gauss] = svmpredict(double(d_te_l), test_data, d_modelGauss);
        [l_predict_label_gauss, l_accuracy_gauss, l_prob_values_gauss] = svmpredict(double(l_te_l), test_data, l_modelGauss);
        
        ac_vect_v(trial) = v_accuracy_gauss(1);
        ac_vect_a(trial) = a_accuracy_gauss(1);
        ac_vect_d(trial) = d_accuracy_gauss(1);
        ac_vect_l(trial) = l_accuracy_gauss(1);
    end
    mean_ac_v(ac) = mean(ac_vect_v); % The mean of accuracies of valence over n trials
    std_ac_v(ac) = std(ac_vect_v); % The std. of accuracies of valence over n trials
    mean_ac_a(ac) = mean(ac_vect_a); % arousal accuracy mean
    std_ac_a(ac) = std(ac_vect_a); % arousal accuracy std.
    mean_ac_d(ac) = mean(ac_vect_d); % dominance accuracy mean
    std_ac_d(ac) = std(ac_vect_d); % dominance accuracy std.
    mean_ac_l(ac) = mean(ac_vect_l); % liking accuracy mean
    std_ac_l(ac) = std(ac_vect_l); % liking accuracy std.
end

mean_ac_v
mean_ac_a
mean_ac_d
mean_ac_l
std_ac_v
std_ac_a
std_ac_d
std_ac_l