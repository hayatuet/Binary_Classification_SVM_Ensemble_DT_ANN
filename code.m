clc; clear; close all;

%% Load the Data:
filename = 'data.csv';
opts = detectImportOptions(filename);
loadNames;
opts.VariableNames = names;
data = readtable(filename, opts);

%% Extracting Input and Output from the table:
% Extracting features as X
X = table2array(data(1:end, 3:end));

% Extracting Labels as y from the table.
y = double((cell2mat(table2array(data(:, {'diagnosis'})))=='M'));

%% Preprocessing
% Batch normalization of the data.
% for making the data mean and scale invariant..
summary(data);
X_mean = mean(X);
X_var  = var(X);

X = (X - X_mean)./X_var;

%%
% k-fold cross validation

rng default;
k = 5;
indices = crossvalind('Kfold', y, k);
accSVM = zeros(1,k);
confmatSVM = zeros(2,2,k);

for i = 1:k
    disp(['SVM Training: Fold: ', num2str(i), ' out of ', num2str(k)]);
    testIdx = (indices==i);
    trainIdx= ~testIdx;
    mdl = fitcsvm(X(trainIdx, :), y(trainIdx));
    trueLab = y(testIdx);
    predLab = predict(mdl, X(testIdx, :));
    accSVM(i) = sum(trueLab == predLab)/length(trueLab);
    confmatSVM(:, :, i) = confusionmat(trueLab, predLab);
end
accSVM_mean = mean(accSVM);
confmatSVM_mean = round(mean(confmatSVM, 3));
figure, plotConf(confmatSVM_mean);

%%
accDT = zeros(1,k);
confmatDT = zeros(2,2,k);

for i = 1:k
    disp(['DecisionTree Training: Fold: ', num2str(i), ' out of ', num2str(k)]);
    testIdx = (indices==i);
    trainIdx= ~testIdx;
    mdl = fitctree(X(trainIdx, :), y(trainIdx));
    trueLab = y(testIdx);
    predLab = predict(mdl, X(testIdx, :));
    accDT(i) = sum(trueLab == predLab)/length(trueLab);
    confmatDT(:, :, i) = confusionmat(trueLab, predLab);
end
accDT_mean = mean(accDT);
confmatDT_mean = round(mean(confmatDT, 3));
figure, plotConf(confmatDT_mean);

%%
accANN = zeros(1,k);
confmatANN = zeros(2,2,k);

for i = 1:k
    disp(['ANN Predicting: Fold: ', num2str(i), ' out of ', num2str(k)]);
    testIdx = (indices==i);
    trueLab = y(testIdx);
    predLab = myANN(X(testIdx, :));
    accANN(i) = sum(trueLab == round(predLab))/length(trueLab);
    confmatANN(:, :, i) = confusionmat(trueLab, round(predLab));
end
accANN_mean = mean(accANN);
confmatANN_mean = round(mean(confmatANN, 3));
figure, plotConf(confmatANN_mean);

%%
accEns = zeros(1,k);
confmatEns = zeros(2,2,k);

for i = 1:k
    disp(['Ensemble Model Training: Fold: ', num2str(i), ' out of ', num2str(k)]);
    testIdx = (indices==i);
    trainIdx= ~testIdx;
    mdl = fitcensemble(X(trainIdx, :), y(trainIdx));
    trueLab = y(testIdx);
    predLab = predict(mdl, X(testIdx, :));
    accEns(i) = sum(trueLab == round(predLab))/length(trueLab);
    confmatEns(:, :, i) = confusionmat(trueLab, round(predLab));
end
accEns_mean = mean(accEns);
confmatEns_mean = round(mean(confmatEns, 3));
figure, plotConf(confmatEns_mean);