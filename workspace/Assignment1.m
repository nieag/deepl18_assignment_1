clc, clear all;
[trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
[valX, valY, valy] = LoadBatch('Dataset/data_batch_5.mat');
[testX, testY, testy] = LoadBatch('Dataset/test_batch.mat');

[d, N] = size(trainX);
[K, ~] = size(trainY);
W = 0.01*randn(K, d);
b = 0.01*randn(K,1);

% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:,1), trainY(:,1), W, b, lambda, 1e-6);
% [grad_W, grad_b] = ComputeGradients(trainX(:, 1), trainY(:, 1), P, W, lambda);
% 
% eps = 1e-6;
% diff_W = CompareGrads(grad_W, ngrad_W)
% diff_b = CompareGrads(grad_b, ngrad_b)

lambda = 1;
GDparams.n_batch=100;
GDparams.eta=0.01;
GDparams.n_epochs = 40;

[Wstar, bstar, trainLoss, valLoss] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
figure;
plot(trainLoss, 'color', 'b'); hold on; grid on;
plot(valLoss, 'color', 'r')
title('Cross entropy loss for training and validation sets');
legend('Training Loss', 'Validation Loss');
xlabel('Epochs')
ylabel('Cross entropy loss')
fname = sprintf('Results/loss_nbatch_%i_eta_%f_n_epochs_%i_lambda_%f.png', GDparams.n_batch, GDparams.eta, GDparams.n_epochs, lambda);
saveas(gcf, fname, 'png');
hold off;
accTest = ComputeAccuracy(testX, testy, Wstar, bstar)


for i=1:10
im = reshape(Wstar(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure;
montage(s_im)
fnameMontage = sprintf('Results/montage_nbatch_%i_eta_%f_n_epochs_%i_lambda_%f.png', GDparams.n_batch, GDparams.eta, GDparams.n_epochs, lambda);
saveas(gcf, fnameMontage, 'png');

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

function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
D = size(X, 2);
Wij = sum(sum(W.^2,1),2);
lcross = -log(sum(Y.*P));
J = (1/D)*sum(lcross)+lambda*Wij;
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
% g = (Y-P)';
% grad_b = -g'/N;
% grad_W = -(g'*X'/N + 2*lambda*W);
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

for i=1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W,lambda);
%         size(grad_W), size(grad_b)
        W = W - eta*grad_W;
        b = b - eta*grad_b;
        
    end
    trainLoss = ComputeCost(trainX, trainY, W, b, lambda);
    tL_saved = [tL_saved;trainLoss];
    valLoss = ComputeCost(valX, valY, W, b, lambda);
    vL_saved = [vL_saved; valLoss];
end

Wstar = W;
bstar = b;
end

%numerical gradients
%slow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end

%quick
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end
