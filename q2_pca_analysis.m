%Loading data and reshaping data
clc; close all; clear all;
load('q2_dataset.mat');

X = reshape(data,[150, 85*125]);

%% Using SVD to apply PCA to our dataset
% =======================================================
columnMean = mean(X,1);
centerX = X - columnMean;

tstart = tic;

[U,S,V] = svd(centerX);
P = U*S;

fprintf("Elapsed time for using SVD to apply PCA: %f\n\n", toc(tstart));

%% Reporting MVEs for reconstructing data with SVD approach
% =======================================================
P2 = centerX*V;
reconstructed_svd_1 = P2*V' + columnMean;
mve1 = sum((X - reconstructed_svd_1).^2, 'all');
fprintf("MVE for using X*V' with SVD approach: %.10f\n", mve1);
reconstructed_svd_2 = P*V' + columnMean;
mve2 = sum((X - reconstructed_svd_2).^2, 'all');
fprintf("MVE for using U*S with SVD approach: %.10f\n\n", mve2);

%% Using Covariance matrix to apply PCA to our dataset
% =======================================================
tstart = tic;

C = cov(centerX);
[Vc,d] = eigs(double(C), 150);
p_c = centerX*Vc;

fprintf("Elapsed time for using covariance to apply PCA: %f\n\n", toc(tstart));
%% Reporting MVE for reconstructing data with Covariance matrix approach
% =======================================================
reconstructed_cov = p_c * Vc' + columnMean;
mve3 = sum((X - reconstructed_cov).^2, 'all');

fprintf("MVE for using X*V' with Covariance matrix approach: %.10f\n\n", mve1);

%% Plotting images for original images, SVD reconstructed images, and covariance matrix reconstructed images
% =======================================================
dataset_images = zeros(3,85,125);
svd_images = zeros(3,85,125);
covariance_images = zeros(3,85,125);

for i = 1:5
    
    subplot(3,5,i); 
        imshow(reshape(data(i,:,:),85,125));
        title(['original image ' num2str(i,'%d')]); 
    subplot(3,5,5+i);    
        imshow(reshape(reconstructed_svd_1(i,:), 85, 125));
        title(['svd image ' num2str(i,'%d')]); 
    subplot(3,5,10+i);
        imshow(reshape(reconstructed_cov(i,:), 85, 125));
        title(['covariance image ' num2str(i,'%d')]); 
end

set(gcf, 'Position', [1400 100 1200 900])
