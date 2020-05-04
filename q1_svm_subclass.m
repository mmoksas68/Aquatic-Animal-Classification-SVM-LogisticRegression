%Loading data and initializing hog data
clc; close all; clear;
load('q1_dataset.mat');

X_hog = [ones(length(hog_features_train), 1) hog_features_train];
y_hog = double(subclass_labels_train);


testX_hog = [ones(length(hog_features_test), 1) hog_features_test];
testy_hog = double(subclass_labels_test);

%% hard margin SVM with polynomial kernel for Hog dataset
% =============================================================
tstart = tic;
[polynomial_parameters_hog, polynomial_hog_accuracies] = cross_validation(X_hog, y_hog, 5, 'polynomial');
temp = templateSVM('KernelFunction','polynomial','PolynomialOrder', polynomial_parameters_hog(1,1),...
                    'KernelScale', polynomial_parameters_hog(1,2), 'Solver' , 'ISDA');
SVMModel = fitcecoc(X_hog,y_hog,'Learners',temp);
p = predict(SVMModel, testX_hog);
fprintf("Performance results for hard margin SVM with polynomial kernel for Hog dataset: \n");
fprintf("Chosen degree: %.4f, gamma value: %.4f\n", polynomial_parameters_hog(1,1), polynomial_parameters_hog(1,2));
[a,c] = report_performance_matrix(p, testy_hog, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% soft margin SVM with radial basis function (rbf) as kernel for Hog dataset
% =============================================================
tstart = tic;
[rbf_soft_parameters_hog, rbf_soft_hog_accuracies] = cross_validation(X_hog, y_hog, 5, 'rbf_soft');

temp = templateSVM('KernelFunction','rbf','BoxConstraint',rbf_soft_parameters_hog(1,1),...
                                        'KernelScale', rbf_soft_parameters_hog(1,2));
SVMModel = fitcecoc(X_hog,y_hog,'Coding','onevsall','Learners',temp);
p = predict(SVMModel, testX_hog);
fprintf("Performance results for soft margin SVM with radial basis function (rbf) as kernel for Hog dataset: \n");
fprintf("Chosen C value: %.4f, gamma value: %.4f\n",rbf_soft_parameters_hog(1,1), rbf_soft_parameters_hog(1,2));
report_performance_matrix(p, testy_hog, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% initializing inception data
% =============================================================
X_inception = [ones(size(inception_features_train, 1), 1) inception_features_train];
y_inception = double(subclass_labels_train);

testX_inception = [ones(size(inception_features_test, 1), 1) inception_features_test];
testy_inception = double(subclass_labels_test);

%% hard margin SVM with polynomial kernel for inception dataset
% =============================================================
tstart = tic;
[polynomial_parameters_inception, polynomial_inception_accuracies] = cross_validation(X_inception, y_inception, 5, 'polynomial');

temp = templateSVM('KernelFunction','polynomial','PolynomialOrder', polynomial_parameters_inception(1,1),...
                    'KernelScale', polynomial_parameters_inception(1,2), 'Solver' , 'ISDA');
SVMModel = fitcecoc(X_inception,y_inception,'Learners',temp);
p = predict(SVMModel, testX_inception);
fprintf("Performance results for hard margin SVM with polynomial kernel for inception dataset: \n");
fprintf("Chosen degree: %.4f, gamma value: %.4f\n", polynomial_parameters_inception(1,1), polynomial_parameters_inception(1,2));
report_performance_matrix(p, testy_inception, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% soft margin SVM with radial basis function (rbf) as kernel for inception dataset
% =============================================================
tstart = tic;
[rbf_soft_parameters_inception, rbf_soft_inception_accuracies] = cross_validation(X_inception, y_inception, 5, 'rbf_soft');

temp = templateSVM('KernelFunction','rbf','BoxConstraint',rbf_soft_parameters_inception(1,1),...
                                        'KernelScale', rbf_soft_parameters_inception(1,2));
SVMModel = fitcecoc(X_inception,y_inception,'Coding','onevsall','Learners',temp);

p = predict(SVMModel, testX_inception);
fprintf("Performance results for soft margin SVM with radial basis function (rbf) as kernel for inception dataset: \n");
fprintf("Chosen C value: %.4f, gamma value: %.4f\n",rbf_soft_parameters_inception(1,1), rbf_soft_parameters_inception(1,2));
report_performance_matrix(p, testy_inception, true);
fprintf("Elapsed time: %f\n\n", toc(tstart));
fprintf("--------------------------------------------------------------\n");

%% =============================================================

% FUNCTIONS
% =============================================================

function [selected_parameters, accuracies] = cross_validation(features, labels, k, function_type)
% Evaluate performance metrics for k fold cross validation for chosen kernel type and
% return best parameters and mean accurasies 
    
    [shuffled_X, shuffled_y, fold_indices] = stratified_k_fold(features, labels, k);
    accuracies_polynomial = zeros(k, 3, 3);
    accuracies_soft_rbf = zeros(k, 3, 3);
    degree = [3, 5, 7];
    C_soft_rbf = [0.01, 1, 100];
    gamma_polynomial = [2^-2, 2, 64];
    gamma_rbf = [2^-2, 2, 64];
    
    for i = 1:k
        first = fold_indices(i,1);
        last = fold_indices(i,2); 
        left_one_X = shuffled_X(first:last, :);
        left_one_y = shuffled_y(first:last, :);        
        current_X = [shuffled_X(1:first-1,:) ; shuffled_X(last+1:size(shuffled_X,1),:)];
        current_y = [shuffled_y(1:first-1,:) ; shuffled_y(last+1:size(shuffled_y,1),:)];
        
        if(strcmp(function_type,'polynomial'))
            for j = 1:length(degree)
                for k = 1:length(gamma_polynomial)
                    temp = templateSVM('KernelFunction','polynomial','PolynomialOrder',degree(1,j),...
                                        'KernelScale', gamma_polynomial(1,k), 'Solver' , 'ISDA');
                    SVMModel = fitcecoc(current_X,current_y,'Learners',temp);
                    p = predict(SVMModel, left_one_X);
                    accuracies_polynomial(i,j,k) = report_performance_matrix(p, left_one_y, false); 
                end                           
            end                             
        end 
        
        if(strcmp(function_type,'rbf_soft'))
            for j = 1:length(C_soft_rbf)
                for k = 1:length(gamma_rbf)
                    temp = templateSVM('KernelFunction','rbf','BoxConstraint',C_soft_rbf(1,j),...
                                        'KernelScale', gamma_rbf(1,k));
                    SVMModel = fitcecoc(current_X,current_y,'Coding','onevsall','Learners',temp);
                    p = predict(SVMModel, left_one_X);
                    accuracies_soft_rbf(i,j,k) = report_performance_matrix(p, left_one_y, false); 
                end                           
            end                             
        end 
    end
    switch function_type
        case 'polynomial'            
            means = mean(accuracies_polynomial);
            accuracies = means;
            max_index = find(means == max(means, [], 'all'), 1);
            [~,b,c] = ind2sub(size(means), max_index);
            selected_parameters = [degree(b), gamma_polynomial(c)];
        case 'rbf_soft'
            means = mean(accuracies_soft_rbf);
            accuracies = means;
            max_index = find(means == max(means, [], 'all'), 1);
            [~,b,c] = ind2sub(size(means), max_index);
            selected_parameters = [C_soft_rbf(b), gamma_rbf(c)];
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

function [accuracy, C] = report_performance_matrix(predictions, labels, isPrinted)

    C = confusionmat(labels, predictions);
    accuracy = sum(diag(C))/size(labels,1);
    
    true_predictions = diag(C)';
    all_classes_predictions = sum(C');
    
    class_seperated_accuracies = true_predictions./all_classes_predictions;
    
    recall = class_seperated_accuracies; 
    precision = true_predictions./sum(C);
    
    macro_recall = sum(recall)/length(recall);
    macro_precision = sum(precision)/length(precision);
    macro_f1 = 2*macro_recall*macro_precision/(macro_recall+macro_precision);
    
    micro_recall = sum(true_predictions)/sum(sum(C'));
    micro_precision = sum(true_predictions)/sum(sum(C));
    micro_f1 = 2*micro_recall*micro_precision/(micro_recall+micro_precision);
    
    if(isPrinted)
        figure
        confusionchart(C);
        for i = 1:10
            fprintf("class %d accuracy: %f\n",i ,class_seperated_accuracies(1,i));
        end
        fprintf("macro recall: %f, macro precision: %f, macro f1: %f\n", macro_recall, macro_precision, macro_f1);
        fprintf("micro recall: %f, micro precision: %f, micro f1: %f\n", micro_recall, micro_precision, micro_f1);
        fprintf("total accuracy: %.4f\n", accuracy);
    end
end
