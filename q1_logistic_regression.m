%Loading data and initializing hog data and weights
clc; close all; clear all;
load('q1_dataset');

X_hog = [ones(length(hog_features_train), 1) hog_features_train];
y_hog = double(superclass_labels_train);

testX_hog = [ones(length(hog_features_test), 1) hog_features_test];
testy_hog = double(superclass_labels_test);

initial_hog = normrnd(0, 0.01, [size(X_hog,2), 1]);

%% training mini batch for hog data
%=============================================================
tstart = tic;

theta_hog = initial_hog;
theta_hog = gradient_ascent_mini_batch(theta_hog, X_hog, y_hog, 25, 0.0001, 1000);

fprintf('Performance Metrics for mini batch Hog: \n');
p = predict(theta_hog, testX_hog);
report_performance_metrics(p, testy_hog);

fprintf("Elapsed time: %f\n\n", toc(tstart));

%% training full batch for hog data
%=============================================================
tstart = tic;
theta_hog = initial_hog;
theta_hog = gradient_ascent_full_batch(theta_hog, X_hog, y_hog, 0.0001, 1000);

fprintf('Performance Metrics for full batch Hog: \n');
p = predict(theta_hog, testX_hog);
report_performance_metrics(p, testy_hog);
fprintf("Elapsed time: %f\n\n", toc(tstart));

%% training stochastic batch for hog data
%=============================================================
tstart = tic;
theta_hog = initial_hog;
% I used mini batch with batch size 1 to apply stochastic 
theta_hog = gradient_ascent_mini_batch(theta_hog, X_hog, y_hog, 1, 0.0001, 1000);

fprintf('Performance Metrics for stochastic Hog: \n');
p = predict(theta_hog, testX_hog);
report_performance_metrics(p, testy_hog);
fprintf("Elapsed time: %f\n\n", toc(tstart));

%% initializing inception data and weights
%=============================================================
X_inception = [ones(size(inception_features_train, 1), 1) inception_features_train];
y_inception = double(superclass_labels_train);

testX_inception = [ones(size(inception_features_test, 1), 1) inception_features_test];
testy_inception = double(superclass_labels_test);

initial_inception = normrnd(0, 0.01, [size(X_inception,2), 1]);

%% training mini batch for inception data
%=============================================================
tstart = tic;
theta_inception = initial_inception;
theta_inception = gradient_ascent_mini_batch(theta_inception, X_inception, y_inception, 25, 0.0001, 1000);

fprintf('Performance Metrics for mini batch Inception: \n');
p = predict(theta_inception, testX_inception);
report_performance_metrics(p, testy_inception);
fprintf("Elapsed time: %f\n\n", toc(tstart));

%% training full batch for inception data
%=============================================================
tstart = tic;
theta_inception = initial_inception;
theta_inception = gradient_ascent_full_batch(theta_inception, X_inception, y_inception, 0.0001, 1000);

fprintf('Performance Metrics for full batch Inception: \n');
p = predict(theta_inception, testX_inception);
report_performance_metrics(p, testy_inception);
fprintf("Elapsed time: %f\n\n", toc(tstart));

%% training stochastic batch for inception data
%=============================================================
tstart = tic;
theta_inception = initial_inception;
% I used mini batch with batch size 1 to apply stochastic 
theta_inception = gradient_ascent_mini_batch(theta_inception, X_inception, y_inception, 1, 0.0001, 1000);

fprintf('Performance Metrics for stochastic Inception: \n');
p = predict(theta_inception, testX_inception);
report_performance_metrics(p, testy_inception);
fprintf("Elapsed time: %f\n\n", toc(tstart));
%% =============================================================

% FUNCTIONS
% =============================================================

function g = sigmoid_one(z)
%computing sigmoid function for each element of matrix z
    g = exp(z)./(1+exp(z));
end

function [final_theta] = gradient_ascent_mini_batch(theta, X, y, batch_size, learning_rate, iteration)
%Compute gradient ascent using mini batch approach (stochastic when batch size is 1)

    grad = zeros(size(theta));

    for i = 1:iteration
        for index = 1:batch_size:size(X, 1)
            current_batch = X(index:index+batch_size-1, :);
            current_batch_y = y(index:index+batch_size-1, :);
            z = current_batch*theta;
            g = sigmoid_one(z);
            grad = learning_rate*(current_batch'*(current_batch_y-g));
            theta = theta + grad;
        end 
    end

    final_theta = theta;

end


function [final_theta] = gradient_ascent_full_batch(theta, X, y, learning_rate, iteration)
%Compute gradient ascent using full batch approach 
    grad = zeros(size(theta));

    for i = 1:iteration
            z = X*theta;
            g = sigmoid_one(z);
            grad = learning_rate*(X'*(y-g));
            theta = theta + grad;
            if (mod(i, 100) == 0)
%                [~,idx] = sort(abs(theta), 'descend');
%                fprintf("The top 10 most important features and their weights for iteration %d are: \n", i);
% 
%                combined = [idx(1:10) theta(idx(1:10), 1)];
%                display(combined);
            end
    end

    final_theta = theta;
end

function p = predict(theta, X)
%predicting giving dataset according to the trained weights

    m = size(X, 1); 
    p = zeros(m, 1);

    results = sigmoid_one(X*theta);
    p(find(results >= 0.5)) = 1;
    p(find(results < 0.5)) = 0;

end

function [accuracy] = report_performance_metrics(predictions, labels)
%printing performance metrics for given predictions of labels. 

    figure
    confusionchart(labels, predictions);

    true_predictions = find(predictions == labels);
    false_predictions = find(predictions ~= labels);
    
    tp = sum(predictions(true_predictions, 1) == 1);
    tn = sum(predictions(true_predictions, 1) == 0);
    fp = sum(predictions(false_predictions, 1) == 1);
    fn = sum(predictions(false_predictions, 1) == 0);
    
    
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    specificity = tn/(tn+fp);
    npv = tn/(tn+fn);
    fpr = fp/(fp+tn);
    fdr = fp/(fp+tp);
    f1 = 2*precision*recall / (precision+recall);
    f2 = 5*precision*recall / ((4*precision) + recall);
    accuracy = (tp+tn)/(tp+tn+fp+fn);    
        
    fprintf("tp: %d, tn: %d, fp: %d, fn: %d, accuracy: %f\n",tp, tn, fp, fn, accuracy);
    fprintf("precision: %.2f, recall: %.2f, specificity: %.2f, negative predictive value: %.2f\n",... 
            precision, recall, specificity, npv);
    fprintf("false positive rate: %.2f, false discovery rate: %.2f, f1: %.2f, f2: %.2f\n\n", fpr, fdr, f1, f2);
end