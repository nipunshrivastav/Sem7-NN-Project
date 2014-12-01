%% Implements PCA followed by Logistic Regression
%% It is assumed that features are extracted and stored as msce_features_matrix variable

%% Visualize
numOfPersons = 32;
input_data = msce_features_matrix;  %Put the data matrix here
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
    tr_percent = 0.8; %Percentage of total data to be used to train
    ac_vect_v = zeros(1,10);
    ac_vect_a = zeros(1,10);
    ac_vect_d = zeros(1,10);
    ac_vect_l = zeros(1,10);
    
    for trial = 1:10
        % Randomly split data into training and test
        index_vector = randperm(numOfPersons*40);
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
        tempo = size(training_data);
        dimen = tempo(2);
        % Train using logistic regression
        v_model = reg(training_data(1:cur_len,:),double(v_tr_l(1:cur_len,:)),dimen);
        a_model = reg(training_data(1:cur_len,:),double(a_tr_l(1:cur_len,:)),dimen);
        d_model = reg(training_data(1:cur_len,:),double(d_tr_l(1:cur_len,:)),dimen);
        l_model = reg(training_data(1:cur_len,:),double(l_tr_l(1:cur_len,:)),dimen);
        % Test the model    
        ac_vect_v(trial) = check_reg(v_model,test_data,double(v_te_l),dimen);
        ac_vect_a(trial) = check_reg(a_model,test_data,double(a_te_l),dimen);
        ac_vect_d(trial) = check_reg(d_model,test_data,double(d_te_l),dimen);
        ac_vect_l(trial) = check_reg(l_model,test_data,double(l_te_l),dimen);
    end
    % The mean and std. of accuracies
    mean_ac_v(ac) = mean(ac_vect_v);
    std_ac_v(ac) = std(ac_vect_v);
    mean_ac_a(ac) = mean(ac_vect_a);
    std_ac_a(ac) = std(ac_vect_a);
    mean_ac_d(ac) = mean(ac_vect_d);
    std_ac_d(ac) = std(ac_vect_d);
    mean_ac_l(ac) = mean(ac_vect_l);
    std_ac_l(ac) = std(ac_vect_l);
end

mean_ac_v
mean_ac_a
mean_ac_d
mean_ac_l
std_ac_v
std_ac_a
std_ac_d
std_ac_l