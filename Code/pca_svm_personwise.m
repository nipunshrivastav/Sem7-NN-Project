%% Implements PCA followed by SVM
%% It is assumed that features are extracted and stored as msce_features_matrix variable
%% Train and test on per person basis

%% Visualize
numOfPersons =5;

%% SVM Train, Predict and Validate

mean_ac_v = zeros(1,length(numOfPersons)); 
std_ac_v = zeros(1,length(numOfPersons)); 
mean_ac_a = zeros(1,length(numOfPersons)); 
std_ac_a = zeros(1,length(numOfPersons));
mean_ac_d = zeros(1,length(numOfPersons)); 
std_ac_d = zeros(1,length(numOfPersons)); 
mean_ac_l = zeros(1,length(numOfPersons)); 
std_ac_l = zeros(1,length(numOfPersons)); 

for person = 1: numOfPersons
    input_data = msce_features_matrix((person-1)*40+1:person*40,:);  %Put the data matrix here
    data = bsxfun(@minus, input_data, mean(input_data)); 
    data = bsxfun(@times, data, 1./std(data)); 
    tempo = size(input_data); 
    m = tempo(1); % Number of patterns

    %% Pre-Processing - PCA
    Sigma = (data'*data) ./ m;
    [U, S, V] = svd(Sigma);
    %% Compressing data - PCA
    k = 4; % Choose the number of dimensions in the output
    U_red = U(:,1:k);
    x_red = zeros(m,k);

    for ni = 1:m
        x_red(ni,:) = (U_red'*data(ni,:)')'; 
    end

    % x_red is the input with 'k' PCA dimensions
    tr_percent = 0.7;
    ac_vect_v = zeros(1,10);
    ac_vect_a = zeros(1,10);
    ac_vect_d = zeros(1,10);
    ac_vect_l = zeros(1,10);
    
    for trial = 1:15 % Number of trials
        % Randomly split data into train and test
        index_vector = randperm(40);
        training_data = zeros(40*tr_percent,k);
        v_tr_l = zeros(40*tr_percent,1);
        a_tr_l= zeros(40*tr_percent,1);
        d_tr_l = zeros(40*tr_percent,1);
        l_tr_l = zeros(40*tr_percent,1);
        test_data = zeros(40*(1-tr_percent),k);
        v_te_l = zeros(40*(1-tr_percent),1);
        a_te_l= zeros(40*(1-tr_percent),1);
        d_te_l = zeros(40*(1-tr_percent),1);
        l_te_l = zeros(40*(1-tr_percent),1);

        for i=1:length(index_vector)
            patternNumber = index_vector(i);
            if i<=tr_percent*length(index_vector)
                training_data(i,:) = x_red(patternNumber,:);
                v_tr_l(i) = valence_labels(patternNumber);
                a_tr_l(i) = arousal_labels(patternNumber);
                d_tr_l(i) = dominance_labels(patternNumber);
                l_tr_l(i) = liking_labels(patternNumber);
            else
                test_data(i-tr_percent*length(index_vector),:) = x_red(patternNumber,:);
                v_te_l(i-tr_percent*length(index_vector)) = valence_labels(patternNumber);
                a_te_l(i-tr_percent*length(index_vector)) =arousal_labels(patternNumber);
                d_te_l(i-tr_percent*length(index_vector)) =dominance_labels(patternNumber);
                l_te_l(i-tr_percent*length(index_vector)) =liking_labels(patternNumber);
            end
        end
        cur_len = floor(length(l_tr_l));
        % train and test
        v_modelGauss = svmtrain(double(v_tr_l(1:cur_len,:)),training_data(1:cur_len,:),'-c 11 -g .004');
        a_modelGauss = svmtrain(double(a_tr_l(1:cur_len,:)),training_data(1:cur_len,:), '-c 11 -g 0.004');
        d_modelGauss = svmtrain(double(d_tr_l(1:cur_len,:)),training_data(1:cur_len,:), '-c 11 -g 0.004');
        l_modelGauss = svmtrain(double(l_tr_l(1:cur_len,:)),training_data(1:cur_len,:), '-c 0.1 -g 0.0025');
        
        [v_predict_label_gauss, v_accuracy_gauss, v_prob_values_gauss] = svmpredict(double(v_te_l), test_data, v_modelGauss);
        [a_predict_label_gauss, a_accuracy_gauss, a_prob_values_gauss] = svmpredict(double(a_te_l), test_data, a_modelGauss);
        [d_predict_label_gauss, d_accuracy_gauss, d_prob_values_gauss] = svmpredict(double(d_te_l), test_data, d_modelGauss);
        [l_predict_label_gauss, l_accuracy_gauss, l_prob_values_gauss] = svmpredict(double(l_te_l), test_data, l_modelGauss);
        
        ac_vect_v(trial) = v_accuracy_gauss(1);
        ac_vect_a(trial) = a_accuracy_gauss(1);
        ac_vect_d(trial) = d_accuracy_gauss(1);
        ac_vect_l(trial) = l_accuracy_gauss(1);
    end
    % Mean and std. of accuracies for a person
    mean_ac_v(person) = mean(ac_vect_v);
    std_ac_v(person) = std(ac_vect_v);
    mean_ac_a(person) = mean(ac_vect_a);
    std_ac_a(person) = std(ac_vect_a);
    mean_ac_d(person) = mean(ac_vect_d);
    std_ac_d(person) = std(ac_vect_d);
    mean_ac_l(person) = mean(ac_vect_l);
    std_ac_l(person) = std(ac_vect_l);
end

mean_ac_v
mean_ac_a
mean_ac_d
mean_ac_l
std_ac_v
std_ac_a
std_ac_d
std_ac_l