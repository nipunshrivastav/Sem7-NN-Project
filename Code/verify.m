function res = verify(w1,w2,test,test_label,test_size)
% Takes input as weight vectors between the input layer and hidden layer - w1
% Takes input as weight vectors between the output layer and hidden layer - w2
% Input features to be predicted upon - test
% Ground truth - test_label
% Number of inputs - test_size

% Returns the accuracy obtained for the input combination by cross checking the predicted output
% with the provided ground truth


res = 0;

for i = 1:test_size
    input_h = test(i,:)*w1;
    output_h = [1 1./(1 + exp(-input_h))];

    input_o = output_h*w2;
    output_o = threshold([1./(1 + exp(-input_o))]);

   a=test_label(i);

    if (isequal(output_o,test_label(i)))
        res = res+1;
    end

end

res = res/test_size*100;