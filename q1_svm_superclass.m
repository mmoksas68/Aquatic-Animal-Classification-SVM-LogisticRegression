%Loading data and initializing hog data
clc; close all; clear;
load('q1_dataset.mat');

X_hog = [ones(length(hog_features_train), 1) hog_features_train];
y_hog = double(superclass_labels_train);

testX_hog = [ones(length(hog_features_test), 1) hog_features_test];
testy_hog = double(superclass_labels_test);

%% soft margin SVM model with linear kernel for Hog dataset
% =============================================================
tstart = tic;
[linear_hog_C, linear_hog_accuracies] = cross_validation(X_hog, y_hog, 5, 'linear');

SVMModel = fitcsvm(X_hog,y_hog,'KernelFunction','linear','BoxConstraint', linear_hog_C);
p = predict(SVMModel, testX_hog);
fprintf("Performance results for soft margin SVM model with linear kernel for Hog dataset: \n");
fprintf("Chosen C value: %.4f\n", linear_hog_C);
report_performance_metrics(p, testy_hog, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% hard margin SVM with radial basis function (rbf) kernel for Hog dataset
% =============================================================
tstart = tic;
[rbf_hog_gamma, rbf_hog_accuracies] = cross_validation(X_hog, y_hog, 5, 'rbf');

SVMModel = fitcsvm(X_hog,y_hog,'KernelFunction','rbf','KernelScale', 1/rbf_hog_gamma);
p = predict(SVMModel, testX_hog);
fprintf("Performance results for hard margin SVM with radial basis function (rbf) kernel for Hog dataset: \n");
fprintf("Chosen gamma value: %.4f\n", rbf_hog_gamma);
report_performance_metrics(p, testy_hog, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% soft margin SVM with radial basis function (rbf) as kernel for Hog dataset
% =============================================================
tstart = tic;
[rbf_soft_parameters, rbf_soft_hog_accuracies] = cross_validation(X_hog, y_hog, 5, 'rbf_soft');

SVMModel = fitcsvm(X_hog,y_hog,'KernelFunction','rbf','BoxConstraint', rbf_soft_parameters(1,1),...
                   'KernelScale', 1/rbf_soft_parameters(1,2));
p = predict(SVMModel, testX_hog);
fprintf("Performance results for soft margin SVM with radial basis function (rbf) as kernel for Hog dataset: \n");
fprintf("Chosen C value: %.4f, gamma value: %.4f\n",rbf_soft_parameters(1,1), rbf_soft_parameters(1,2));
report_performance_metrics(p, testy_hog, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% initializing inception data
% =============================================================
X_inception = [ones(size(inception_features_train, 1), 1) inception_features_train];
y_inception = double(superclass_labels_train);

testX_inception = [ones(size(inception_features_test, 1), 1) inception_features_test];
testy_inception = double(superclass_labels_test);

%% soft margin SVM model with linear kernel for inception dataset
% =============================================================
tstart = tic;
[linear_inception_C, linear_inception_accuracies] = cross_validation(X_inception, y_inception, 5, 'linear');

SVMModel = fitcsvm(X_inception,y_inception,'KernelFunction','linear','BoxConstraint', linear_inception_C);
p = predict(SVMModel, testX_inception);
fprintf("Performance results for soft margin SVM model with linear kernel for inception dataset: \n");
fprintf("Chosen C value: %.4f\n", linear_inception_C);
report_performance_metrics(p, testy_inception, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% hard margin SVM with radial basis function (rbf) kernel for inception dataset
% =============================================================
tstart = tic;
[rbf_inception_gamma, rbf_inception_accuracies] = cross_validation(X_inception, y_inception, 5, 'rbf');

SVMModel = fitcsvm(X_inception,y_inception,'KernelFunction','rbf','KernelScale', 1/rbf_inception_gamma);
p = predict(SVMModel, testX_inception);
fprintf("Performance results for hard margin SVM with radial basis function (rbf) kernel for inception dataset: \n");
fprintf("Chosen gamma value: %.4f\n", rbf_inception_gamma);
report_performance_metrics(p, testy_inception, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% soft margin SVM with radial basis function (rbf) as kernel for inception dataset
% =============================================================
tstart = tic;
[rbf_soft_parameters, rbf_soft_inception_accuracies] = cross_validation(X_inception, y_inception, 5, 'rbf_soft');

SVMModel = fitcsvm(X_inception,y_inception,'KernelFunction','rbf','BoxConstraint', rbf_soft_parameters(1,1),...
                   'KernelScale', 1/rbf_soft_parameters(1,2));
p = predict(SVMModel, testX_inception);
fprintf("Performance results for soft margin SVM with radial basis function (rbf) as kernel for inception dataset: \n");
fprintf("Chosen C value: %.4f, gamma value: %.4f\n",rbf_soft_parameters(1,1), rbf_soft_parameters(1,2));
report_performance_metrics(p, testy_inception, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");
%% =============================================================

% FUNCTIONS
% =============================================================

function [selected_parameters, accuracies] = cross_validation(features, labels, k, kernel_type)
% Evaluate performance metrics for k fold cross validation for chosen kernel type and
% return best parameters and mean accurasies 

    [shuffled_X, shuffled_y, fold_indices] = stratified_k_fold(features, labels, k);
    accuracies_linear = zeros(k, 5);
    accuracies_rbf = zeros(k, 7);
    accuracies_soft_rbf = zeros(k, 3, 3);
    C_linear = [0.01, 0.1, 1, 10, 100];
    gamma_rbf = [2^-4, 2^-3, 2^-2, 2^-1, 1, 2, 64];
    C_soft_rbf = [0.01, 1, 100];
    gamma_soft_rbf = [2^-2, 2, 64];
    
    for i = 1:k
        first = fold_indices(i,1);
        last = fold_indices(i,2); 
        left_one_X = shuffled_X(first:last, :);
        left_one_y = shuffled_y(first:last, :);        
        current_X = [shuffled_X(1:first-1,:) ; shuffled_X(last+1:size(shuffled_X,1),:)];
        current_y = [shuffled_y(1:first-1,:) ; shuffled_y(last+1:size(shuffled_y,1),:)];
        
        if(strcmp(kernel_type,'linear'))
            for j = 1:length(C_linear)
                SVMModel = fitcsvm(current_X,current_y,'KernelFunction','linear',...
                                   'BoxConstraint',C_linear(1,j));
                p = predict(SVMModel, left_one_X);
                accuracies_linear(i, j) = report_performance_metrics(p, left_one_y, false);
            end
        end
        
        if(strcmp(kernel_type,'rbf'))
            for j = 1:length(gamma_rbf)
                SVMModel = fitcsvm(current_X,current_y,'KernelFunction','rbf',...
                                   'KernelScale', 1/gamma_rbf(1,j));
                p = predict(SVMModel, left_one_X);
                accuracies_rbf(i, j) = report_performance_metrics(p, left_one_y, false);    
            end                  
        end
        
        if(strcmp(kernel_type,'rbf_soft'))
            for j = 1:length(C_soft_rbf)
                for k = 1:length(gamma_soft_rbf)
                    SVMModel = fitcsvm(current_X,current_y,'KernelFunction','rbf',...
                                        'BoxConstraint',C_soft_rbf(1,j),'KernelScale', 1/gamma_soft_rbf(1,k));
                    p = predict(SVMModel, left_one_X);
                    accuracies_soft_rbf(i,j,k) = report_performance_metrics(p, left_one_y, false); 
                end                           
            end                             
        end 
    end
    
    switch kernel_type
        case 'linear'
            means = mean(accuracies_linear);
            accuracies = means;
            max_index = find(means == max(means), 1);
            selected_parameters = C_linear(max_index);
        case 'rbf'            
            means = mean(accuracies_rbf);
            accuracies = means;
            max_index = find(means == max(means), 1);
            selected_parameters = gamma_rbf(max_index);
        case 'rbf_soft'
            means = mean(accuracies_soft_rbf);
            accuracies = means;
            max_index = find(means == max(means, [], 'all'), 1);
            [a,b,c] = ind2sub(size(means), max_index);
            selected_parameters = [C_soft_rbf(b), gamma_soft_rbf(c)];
    end
end

function [splitted_data_X, splitted_data_y, k_indices] = stratified_k_fold(features, labels, k)
% Splits the given data into k folds and returns data and indices

    classification_labels = [];
    max_length = 0;
    for i = 1 : size(labels, 1)
        if(isempty(find(classification_labels == labels(i,1), 1)))
            classification_labels = [classification_labels labels(i,1)];
            max_length = max(sum(labels == labels(i, 1)), max_length);
        end
    end
    classification_indices = zeros(max_length, size(classification_labels, 2));
    for i = 1 : size(classification_labels, 2)
        indices = find(labels == classification_labels(1, i));
        classification_indices(1:length(indices), i) = indices;
    end
    
    for i = 1 : size(classification_indices, 2)
        shuffle = randperm(size(classification_indices, 1));
        classification_indices(:, i) = classification_indices(shuffle, i); 
    end
    
    shuffled_indices = zeros(length(labels), 1);
    count = 1;
    for i = 1 : size(classification_indices, 1)
        for j = 1 : size(classification_indices, 2)
            if(classification_indices(i, j) ~= 0)
                shuffled_indices(count, 1) = classification_indices(i, j);
                count = count + 1;
            end    
        end
    end
    k_indices = zeros(k, 2);
    one_fold_size = ceil(size(labels, 1)/k);
    for i = 1 : k
         k_indices(i, 1) = one_fold_size * (i-1) + 1;  
         k_indices(i, 2) = min(k_indices(i, 1) + one_fold_size - 1, size(labels, 1));  
    end
    splitted_data_y = labels(shuffled_indices,1);
    splitted_data_X = features(shuffled_indices, :);
end

function [accuracy] = report_performance_metrics(predictions, labels, isPrinted)
%printing performance metrics for given predictions of labels. 

    true_predictions = find(predictions == labels);
    false_predictions = find(predictions ~= labels);
    
    tp = sum(predictions(true_predictions, 1) == 1);
    tn = sum(predictions(true_predictions, 1) == 0);
    fp = sum(predictions(false_predictions, 1) == 1);
    fn = sum(predictions(false_predictions, 1) == 0);
    
    
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    accuracy = (tp+tn)/(tp+tn+fp+fn);
    
    if(isPrinted)
        figure
        confusionchart(labels, predictions);
    
        fprintf("accuracy: %f, precision: %.2f, recall: %.2f\n\n", accuracy, precision, recall);
    end    
end
