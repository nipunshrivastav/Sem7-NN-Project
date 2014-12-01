Feature Extraction:

Firstly, extract features or load final_workspace.mat which contains a backup

To extract features:
1) Download DEAP dataset and store it in code folder
2) Run psd_hamming_overlap.m to extract psd features. These are stored in a variable 'feature_vector'
3) Run 'msce_features_matrix.m' to extract MSCE features. These are stored in a variable ' msce_features_matrix'
(Note: msce_features_matrix.m requires MATLAB 2013a or plus and many hours on 8GB machine)

Alternatively, load 'final_workspace.mat' which has the 'feature_vector' and 'msce_features_matrix' the variables

Running SVM with PCA and RBF Kernel and MSCE Features:
Install 'libsvm' and set path
Run pca_svm.m
The gamma, c values and number of PCA dimensions can be altered inside pca_svm.m 
It performs 10 trials by default and reports mean and standard deviation of accuracies on console

Running Neural Network Classifier with PCA:
Run pca_neural.m
This uses neural_classifier.m, verify.m and threshold.m
The number of number of hidden nodes can be altered inside neural_classifier.m
The number of PCA dimensions and number of trials can be changed in pca_neural.m
The mean and standard deviation of accuracy is reported on console

Running Logistic Regression: 
Run pca_logistic.m
This uses reg.m and check_reg.m
The stopping threshold for logistic regression can be changed from inside reg.m
The number of PCA dimensions and number of trials can be changed in pca_logistic.m
The mean and standard deviation of accuracy is reported on console

Running SVM with PCA and MSCE Features on per person basis:
Run pca_svm_personwise.m
Number of persons, PCA dimensions, gamma and c values can be changed inside pca_svm_personwise.m
Reports a vector of mean of accuracies for each person
Reports a vector of std. of accuracies for each person

Running Neural Network with PCA and MSCE Features on per person basis:
Run pca_neural_personwise.m
The number of number of hidden nodes can be altered inside neural_classifier.m
Number of persons, PCA dimensions can be changed inside pca_svm_personwise.m
Reports a vector of mean of accuracies for each person
Reports a vector of std. of accuracies for each person

Running Logistic Regression with PCA and MSCE Features on per person basis:
Run pca_logistic_personwise.m
This uses reg.m and check_reg.m
This uses neural_classifier.m, verify.m and threshold.m
The stopping threshold for logistic regression can be changed from inside reg.m
Number of persons, PCA dimensions can be changed inside pca_svm_personwise.m
Reports a vector of mean of accuracies for each person
Reports a vector of std. of accuracies for each person



