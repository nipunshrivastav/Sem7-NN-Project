function res = threshold(arr)
% Simply applies a threshold of 0.5on the output of logistic regression
% to convert the output to binary value instead of having it somewhere between 0 and 1

res = arr - arr;

for i = 1:length(arr)
    if (arr(i)>0.5)
        res(i) = 1;
    end
end