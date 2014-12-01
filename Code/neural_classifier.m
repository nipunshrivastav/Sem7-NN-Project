function accuracy = neural_classifier(train, train_label, test, test_label)

    % takes input as the test set, test set label, train set and train set label
    % learns a neural network using the training set and label and calls a verify
    % function eventually to find the accuracy of the leearnt network

    test_size = size(test,1);
    train_size = size(train,1);

    i = size(train,2);% input
    h = 10; %hidden_layer_neurons
    o = 1; %output
    eta = 1; % Learning Rate

    w1 = 2 * rand(i+1,h) - 1;% Weights and Bias between input layer and hidden layer
    w2 = 2 * rand(h+1,o) - 1;% Weights and Bias between hidden layer and output layer


    d_w1 = zeros(i+1,h);
    d_w2 = zeros(h+1,o);


    output_h = [1 zeros(1,h)]; %output at hidden layer
    output_o = zeros(1,o); %output at output layer

    d_h = zeros(1,h+1);% slope and error product at hidden
    d_o = zeros(1,o); % slope and error product at output

    error = 2;
    round = 0;
    iter = 0;
    errorNew = 0;


    test = double([(ones(test_size,1)) test]);
    train = double([(ones(train_size,1)) train]);

    while (error > 0.01 || round<85)
         error = 0;
         round = round+1;

         iter = iter+1;
         eta = 1/sqrt(iter);

         errorOld = errorNew;
         errorNew = 0;

         a = randperm(train_size);


         for k = 1:train_size


             input_h = train(a(k),:)*w1;
             output_h = [1 1./(1 + exp(-input_h))];
             input_o = output_h*w2;
             output_o = [1./(1 + exp(-input_o))];

             err = train_label(a(k)) - output_o;
             d_o = output_o.*(1-output_o).*err;

             for j = 1:h+1
                 d_w2(j,:) = d_o*output_h(j);
             end

             w2 = w2 + eta * d_w2;


             for j = 2:h+1
                 d_h(j) = (d_o*w2(j,:)')*output_h(j)*(1-output_h(j));
             end

             for j = 1:i+1
                 d_w1(j,:) = d_h(2:h+1)*train(a(k),j);
             end

             w1 = w1 + eta * d_w1;

             z = sum(err.^2);
             errorNew = errorNew + z;

     end

     error = abs(errorOld - errorNew);
     %pause;

    end
    accuracy = verify(w1,w2,test,test_label',test_size);