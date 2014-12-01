%% Implements PCA followed by Logistic Regression
%% It is assumed that features are extracted and stored as msce_features_matrix variable
%% Train and test for each person separately
%% Visualize
numOfPersons = 5;



%% SVM Train, Predict and Validate

mean_ac_v = zeros(1,length(numOfPersons)); 
std_ac_v = zeros(1,length(numOfPersons)); 
mean_ac_a = zeros(1,length(numOfPersons)); 
std_ac_a = zeros(1,length(numOfPersons));
mean_ac_d = zeros(1,length(numOfPersons)); 
std_ac_d = zeros(1,length(numOfPersons)); 
mean_ac_l = zeros(1,length(numOfPersons)); 
std_ac_l = zeros(1,length(numOfPersons)); 

for person = 1: numOfPersons % Find mean and std. of accuracies for each person separately
    input_data = msce_features_matrix((person-1)*40+1:person*40,:);  %Put the data matrix here
    data = bsxfun(@minus, input_data, mean(input_data)); 
    data = bsxfun(@times, data, 1./std(data)); 
    tempo = size(input_data); 
    m = tempo(1); % Number of patterns

    %% Pre-Processing
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
    
    tr_percent = 0.8; %Percentage of total data to be used to train
    ac_vect_v = zeros(1,10);
    ac_vect_a = zeros(1,10);
    ac_vect_d = zeros(1,10);
    ac_vect_l = zeros(1,10);
    
    for trial = 1:10 %Number of trials
        % Randomly split data into training and test
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
        cur_len = floor(length(l_tr_l));
        tempo = size(training_data);
        dimen = tempo(2);
        % Train
        v_model = reg(training_data(1:cur_len,:),double(v_tr_l(1:cur_len,:)),dimen);
        a_model = reg(training_data(1:cur_len,:),double(a_tr_l(1:cur_len,:)),dimen);
        d_model = reg(training_data(1:cur_len,:),double(d_tr_l(1:cur_len,:)),dimen);
        l_model = reg(training_data(1:cur_len,:),double(l_tr_l(1:cur_len,:)),dimen);
        %Test
        ac_vect_v(trial) = check_reg(v_model,test_data,double(v_te_l),dimen);
        ac_vect_a(trial) = check_reg(a_model,test_data,double(a_te_l),dimen);
        ac_vect_d(trial) = check_reg(d_model,test_data,double(d_te_l),dimen);
        ac_vect_l(trial) = check_reg(l_model,test_data,double(l_te_l),dimen);
    end
    % Mean and std. of accuracies
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