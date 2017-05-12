%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

% Choose some alpha value
alpha = 0.01;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history1] = gradientDescentMulti(X, y, theta, alpha, num_iters);

alpha = 0.03;
theta = zeros(3, 1);
[theta, J_history2] = gradientDescentMulti(X, y, theta, alpha, num_iters);

alpha = 0.1;
theta = zeros(3, 1);
[theta, J_history3] = gradientDescentMulti(X, y, theta, alpha, num_iters);

alpha = 0.3;
theta = zeros(3, 1);
[theta, J_history4] = gradientDescentMulti(X, y, theta, alpha, num_iters);

alpha = 1;
theta = zeros(3, 1);
[theta, J_history5] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history1), J_history1, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold on
plot(1:numel(J_history2), J_history2, '-r', 'LineWidth', 2);
plot(1:numel(J_history3), J_history3, '-g', 'LineWidth', 2);
plot(1:numel(J_history4), J_history4, '-k', 'LineWidth', 2);
plot(1:numel(J_history5), J_history5, '-c', 'LineWidth', 2);