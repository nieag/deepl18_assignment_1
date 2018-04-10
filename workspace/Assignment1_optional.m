clc, clear all;
rng(400);
%%% Load data
% [trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
% [valX, valY, valy] = LoadBatch('Dataset/data_batch_5.mat');
[testX, testY, testy] = LoadBatch('Dataset/test_batch.mat');

%%% Use all data sets
[tx1, tY1, ty1] = LoadBatch('Dataset/data_batch_1.mat');
[tx2, tY2, ty2] = LoadBatch('Dataset/data_batch_2.mat');
[tx3, tY3, ty3] = LoadBatch('Dataset/data_batch_3.mat');
[tx4, tY4, ty4] = LoadBatch('Dataset/data_batch_4.mat');
[tx5, tY5, ty5] = LoadBatch('Dataset/data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('Dataset/test_batch.mat');

X_train = [tx1, tx2, tx3, tx4, tx5(:, 1:9000)];
Y_train = [tY1, tY2, tY3, tY4, tY5(:, 1:9000)];
y_train = [ty1, ty2, ty3, ty4, ty5(:, 1:9000)];

X_valid = tx5(:,9001:10000);
Y_valid = tY5(:,9001:10000);
y_valid = ty5(:,9001:10000);

%%% Initialise weights and biases
[d, N] = size(X_train);
[K, ~] = size(Y_train);
W = 0.01*randn(K, d);
b = 0.01*randn(K,1);
lambda = 0.1;
GDparams.loss = 'crossEnt';
% J1 = ComputeCost(trainX, trainY, W, b, lambda, GDparams)


%%% Gradient evaluation
% P = EvaluateClassifier(trainX(:,1), W, b);
% 
% [ngrad_b_slow, ngrad_W_slow] = ComputeGradsNumSlow(trainX(:,1), trainY(:,1), W, b, lambda, 1e-6, GDparams);
% [ngrad_b_quick, ngrad_W_quick] = ComputeGradsNum(trainX(:,1), trainY(:, 1), W, b, lambda, 1e-6, GDparams);
% [grad_W, grad_b] = ComputeGradients(trainX(:, 1), trainY(:, 1), P, W, lambda, GDparams);
% 
% eps = 1e-6;
% diff_W_slow = CompareGrads(grad_W, ngrad_W_slow)
% diff_b_slow = CompareGrads(grad_b, ngrad_b_slow)
% 
% diff_W_quick = CompareGrads(grad_W, ngrad_W_quick)
% diff_b_quick = CompareGrads(grad_b, ngrad_b_quick)

%%% Train and evaluate network on test data
lambda = 0.01;
GDparams.n_batch=100;
GDparams.eta=0.02;
GDparams.n_epochs = 200;
GDparams.decay = 0.9;


[Wstar, bstar, trainLoss, valLoss] = MiniBatchGD(X_train, Y_train, X_valid, Y_valid, GDparams, W, b, lambda);
figure;
plot(trainLoss, 'color', 'b'); hold on; grid on;
plot(valLoss, 'color', 'r')
title('Cross entropy loss for training and validation sets');
legend('Training Loss', 'Validation Loss');
xlabel('Epochs')
ylabel('Cross entropy loss')
% fname = sprintf('Results/loss_nbatch_%i_eta_%f_n_epochs_%i_lambda_%f.png', GDparams.n_batch, GDparams.eta, GDparams.n_epochs, lambda);
% saveas(gcf, fname, 'png');
hold off;

accTest = ComputeAccuracy(X_test, y_test, Wstar, bstar)

%%% Visualise weight matrices
for i=1:10
im = reshape(Wstar(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure;
montage(s_im)
% fnameMontage = sprintf('Results/all_data_montage_nbatch_%i_eta_%f_n_epochs_%i_lambda_%f.png', GDparams.n_batch, GDparams.eta, GDparams.n_epochs, lambda);
% saveas(gcf, fnameMontage, 'png');

% Sub-functions
function [X, Y, y] = LoadBatch(filename)
dataSet = load(filename);
X = double(dataSet.data)'/255;
y = double(dataSet.labels+1)';
N = length(y);
K = max(y);
Y = zeros(K, N);
for i = 1:N
    Y(y(i), i) = 1;
end
end

function P = EvaluateClassifier(X, W, b)
sum = bsxfun(@plus, W*X, b);
P = softmax(sum);
end

function J = ComputeCost(X, Y, W, b, lambda, GDparams)
P = EvaluateClassifier(X, W, b);
D = size(X, 2);
Wij = sum(sum(W.^2,1),2);
if strcmp(GDparams.loss, 'crossEnt')
    loss = -log(sum(Y.*P));
elseif strcmp(GDparams.loss, 'SVM')
    temp = max(0, P - P.*Y + 1);
    loss = sum(temp)-1;
end
J = (1/D)*sum(loss)+lambda*Wij;
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~, kStar] = max(P);
correct = kStar==y;
acc = sum(correct)/length(correct);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
N = size(X,2);
k = size(Y,1);
grad_b= zeros(k,1);
grad_W = zeros(size(W));
for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    grad_b = grad_b + g';
    grad_W = grad_W + g'*X(:,i)';
end
grad_b = grad_b/N;
grad_W = grad_W/N + 2*lambda*W;
end

function RelError = CompareGrads(ga, gn)
RelError = max(max(abs(ga-gn)));
end

function [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda)
n_batch = GDparams.n_batch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs;
N = size(trainX,2);
tL_saved=[];
vL_saved=[];
decay = GDparams.decay;

valLossOld = 10000;
for i=1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W,lambda);
        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end
    if mod(i,10)==0
    eta = eta * decay;    
    end
    trainLoss = ComputeCost(trainX, trainY, W, b, lambda, GDparams);
    valLoss = ComputeCost(valX, valY, W, b, lambda, GDparams)
    if valLoss<=valLossOld
       Wstar = W;
       bstar = b;
    end
    tL_saved = [tL_saved;trainLoss];
    vL_saved = [vL_saved; valLoss];
    valLossOld=valLoss;
end

% Wstar = W;
% bstar = b;
end

%numerical gradients
%slow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h, GDparams)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda, GDparams);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda, GDparams);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda, GDparams);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda, GDparams);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end

%quick
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h, GDparams)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda, GDparams);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda, GDparams);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda, GDparams);
    
    grad_W(i) = (c2-c) / h;
end
end
