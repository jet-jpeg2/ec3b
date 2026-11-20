function [cfg, tunedMdls, HyperparamTable] = trainModels(D, seed)
% trainModels tunes 1 baseline and 5 classifiers using 5-fold cv,
% saves tuned hyperparameters and trained models, and returns cfg.

if nargin < 2 || isempty(seed), seed = 42; end
if nargin < 1 || isempty(D), D = prepData("viral_songs.csv", 'Seed', seed); end
if ~isfolder("models"), mkdir("models"); end

rng(seed, 'twister'); % for reproducibility

X_train = D.Xtrain{:, :};
Y_train = D.ytrain;
cv      = D.cv;

%% ==================== TASK 1: HYPERPARAMETER TUNING (5-FOLD CV) ====================
fprintf('--- Task 1: Tuning Models (5-Fold CV) ---\n');

% set bayesian optimization options
hpOpts.CVPartition             = cv;
hpOpts.AcquisitionFunctionName = 'expected-improvement-plus';
hpOpts.MaxObjectiveEvaluations = 10;
hpOpts.ShowPlots               = false;
hpOpts.Verbose                 = 0;

% 1. logistic regression (baseline)
rng(seed, 'twister');
mdl_lr = fitclinear(X_train, Y_train, 'Learner', 'logistic', ...
    'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
    'HyperparameterOptimizationOptions', hpOpts);
fprintf('1. Logistic Regression tuned.\n');

% 2. naive bayes
rng(seed, 'twister');
paramsNB = hyperparameters('fitcnb', X_train, Y_train);
for i = 1:numel(paramsNB), paramsNB(i).Optimize = false; end
paramsNB = setParam(paramsNB, 'DistributionNames', []);
paramsNB = setParam(paramsNB, 'Width', [0.30, 1.50]);

hpOptsNB = hpOpts;
hpOptsNB.MaxObjectiveEvaluations = 2;
mdl_nb = fitcnb(X_train, Y_train, ...
    'OptimizeHyperparameters', paramsNB, ...
    'HyperparameterOptimizationOptions', hpOptsNB);
fprintf('2. Naive Bayes tuned.\n');

% 3. decision tree
rng(seed, 'twister');
mdl_tree = fitctree(X_train, Y_train, ...
    'OptimizeHyperparameters', {'MaxNumSplits', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', hpOpts);
fprintf('3. Decision Tree tuned.\n');

% 4. random forest (bagging ensemble)
rng(seed, 'twister');
paramsRF = hyperparameters('fitcensemble', X_train, Y_train, 'Tree');
for i = 1:numel(paramsRF), paramsRF(i).Optimize = false; end
paramsRF = setParam(paramsRF, 'NumLearningCycles', [20, 60]);
paramsRF = setParam(paramsRF, 'MinLeafSize', [10, 40]);
paramsRF = setParam(paramsRF, 'NumVariablesToSample', [3, 10]);

mdl_rf = fitcensemble(X_train, Y_train, 'Method', 'Bag', ...
    'OptimizeHyperparameters', paramsRF, ...
    'HyperparameterOptimizationOptions', hpOpts);
fprintf('4. Random Forest tuned.\n');

% 5. gradient boosted trees (boosting ensemble)
rng(seed, 'twister');
paramsGB = hyperparameters('fitcensemble', X_train, Y_train, 'Tree');
for i = 1:numel(paramsGB), paramsGB(i).Optimize = false; end
paramsGB = setParam(paramsGB, 'NumLearningCycles', [20, 60]);
paramsGB = setParam(paramsGB, 'LearnRate', [0.03, 0.25]);
paramsGB = setParam(paramsGB, 'MaxNumSplits', [2, 12]);

mdl_gb = fitcensemble(X_train, Y_train, 'Method', 'LogitBoost', ...
    'OptimizeHyperparameters', paramsGB, ...
    'HyperparameterOptimizationOptions', hpOpts);
fprintf('5. Gradient Boosted Trees tuned.\n');

% 6. neural network
rng(seed, 'twister');
paramsNN = hyperparameters('fitcnet', X_train, Y_train);
for i = 1:numel(paramsNN), paramsNN(i).Optimize = false; end
paramsNN = setParam(paramsNN, 'Layer_1_Size', [8, 64]);
paramsNN = setParam(paramsNN, 'Activations', []);
paramsNN = setParam(paramsNN, 'Lambda', [1e-5, 1e-1]);

mdl_nn = fitcnet(X_train, Y_train, 'Standardize', false, 'IterationLimit', 120, ...
    'OptimizeHyperparameters', paramsNN, ...
    'HyperparameterOptimizationOptions', hpOpts);
fprintf('6. Neural Network tuned.\n\n');

tunedMdls = {mdl_lr; mdl_nb; mdl_tree; mdl_rf; mdl_gb; mdl_nn};

%% ==================== TASK 2: EXTRACT SELECTED HYPERPARAMETERS ====================
fprintf('--- Task 2: Selected Hyperparameters ---\n');

% logistic regression parameters ('ridge (l2)' -> 'ridge', 'lasso (l1)' -> 'lasso')
bestLambdaLR = mdl_lr.Lambda;
if contains(lower(char(mdl_lr.Regularization)), 'lasso')
    bestRegLR = 'lasso';
else
    bestRegLR = 'ridge';
end

% naive bayes parameters
bestNB = mdl_nb.HyperparameterOptimizationResults.XAtMinObjective;
if strcmp(char(bestNB.DistributionNames), 'kernel') && ~isnan(bestNB.Width)
    nbWidth   = bestNB.Width;
    nbSummary = sprintf("DistributionNames='kernel', Width=%.4f", nbWidth);
else
    nbSummary = "DistributionNames='normal'";
end

% decision tree, random forest, gradient boosting, and neural network parameters
bestDT = mdl_tree.HyperparameterOptimizationResults.XAtMinObjective;
bestRF = mdl_rf.HyperparameterOptimizationResults.XAtMinObjective;
bestGB = mdl_gb.HyperparameterOptimizationResults.XAtMinObjective;
bestNN = mdl_nn.HyperparameterOptimizationResults.XAtMinObjective;

if ~isnan(bestRF.NumVariablesToSample)
    rfSummary = sprintf("NumLearningCycles=%d, MinLeafSize=%d, NumVariablesToSample=%d", ...
        bestRF.NumLearningCycles, bestRF.MinLeafSize, bestRF.NumVariablesToSample);
else
    rfSummary = sprintf("NumLearningCycles=%d, MinLeafSize=%d, NumVariablesToSample=default (5)", ...
        bestRF.NumLearningCycles, bestRF.MinLeafSize);
end

nnLayerSize = mdl_nn.LayerSizes(1);
nnAct       = char(mdl_nn.Activations);
nnLambda    = bestNN.Lambda;

% build summary table for paper table iii
modelNames = {'Logistic Regression'; 'Naive Bayes'; 'Decision Tree'; ...
              'Random Forest'; 'Gradient Boosted Trees'; 'Neural Network'};
modelFamilies = {'Linear (Baseline)'; 'Probabilistic'; 'Tree'; ...
                 'Ensemble (Bagging)'; 'Ensemble (Boosting)'; 'Neural Network'};

selectedParams = [ ...
    sprintf("Regularization='%s', Lambda=%.4g", bestRegLR, bestLambdaLR); ...
    nbSummary; ...
    sprintf("MaxNumSplits=%d, MinLeafSize=%d", bestDT.MaxNumSplits, bestDT.MinLeafSize); ...
    rfSummary; ...
    sprintf("NumLearningCycles=%d, LearnRate=%.4f, MaxNumSplits=%d", ...
            bestGB.NumLearningCycles, bestGB.LearnRate, bestGB.MaxNumSplits); ...
    sprintf("LayerSizes=%d, Activations='%s', Lambda=%.4g", ...
            nnLayerSize, nnAct, nnLambda)];

HyperparamTable = table(string(modelNames), string(modelFamilies), selectedParams, ...
    'VariableNames', {'Model', 'Family', 'FinalTunedHyperparameters'});

disp(HyperparamTable);
writetable(HyperparamTable, fullfile("models", "tuned_hyperparameters.csv"));

%% ==================== TASK 3: PACKAGE MODELS FOR EVALUATION ====================
fprintf('--- Task 3: Saving Trained Models ---\n');

cfg.dataFile     = 'viral_songs.csv';
cfg.outDir       = 'results';
cfg.seed         = seed;
cfg.selectMetric = 'F1';

% 1. logistic regression recipe
cfg.models(1).Name = 'Logistic Regression';
cfg.models(1).Fit  = @(X, y) fitclinear(X, y, 'Learner', 'logistic', ...
    'Regularization', bestRegLR, 'Lambda', bestLambdaLR);

% 2. naive bayes recipe
cfg.models(2).Name = 'Naive Bayes';
if strcmp(char(bestNB.DistributionNames), 'kernel') && ~isnan(bestNB.Width)
    cfg.models(2).Fit = @(X, y) fitcnb(X, y, 'DistributionNames', 'kernel', 'Width', nbWidth);
else
    cfg.models(2).Fit = @(X, y) fitcnb(X, y, 'DistributionNames', 'normal');
end

% 3. decision tree recipe
cfg.models(3).Name = 'Decision Tree';
cfg.models(3).Fit  = @(X, y) fitctree(X, y, 'MaxNumSplits', bestDT.MaxNumSplits, ...
    'MinLeafSize', bestDT.MinLeafSize);

% 4. random forest recipe
if ~isnan(bestRF.NumVariablesToSample)
    tRF = templateTree('MinLeafSize', bestRF.MinLeafSize, ...
                       'NumVariablesToSample', bestRF.NumVariablesToSample);
else
    tRF = templateTree('MinLeafSize', bestRF.MinLeafSize);
end
cfg.models(4).Name = 'Random Forest';
cfg.models(4).Fit  = @(X, y) fitcensemble(X, y, 'Method', 'Bag', ...
    'NumLearningCycles', bestRF.NumLearningCycles, 'Learners', tRF);

% 5. gradient boosted trees recipe
tGB = templateTree('MaxNumSplits', bestGB.MaxNumSplits);
cfg.models(5).Name = 'Gradient Boosted Trees';
cfg.models(5).Fit  = @(X, y) fitcensemble(X, y, 'Method', 'LogitBoost', ...
    'NumLearningCycles', bestGB.NumLearningCycles, ...
    'LearnRate', bestGB.LearnRate, 'Learners', tGB);

% 6. neural network recipe
cfg.models(6).Name = 'Neural Network';
cfg.models(6).Fit  = @(X, y) fitcnet(X, y, 'Standardize', false, 'IterationLimit', 120, ...
    'LayerSizes', nnLayerSize, 'Activations', nnAct, ...
    'Lambda', nnLambda);

% fit final models on full training set and save
trainedModels = cell(6, 1);
for i = 1:6
    rng(seed, 'twister');
    trainedModels{i} = cfg.models(i).Fit(X_train, Y_train);
end

prepParams = D.Params;
save(fullfile("models", "trained_models.mat"), 'trainedModels', 'tunedMdls', ...
    'HyperparamTable', 'modelNames', 'modelFamilies', 'prepParams');
fprintf('Saved models/tuned_hyperparameters.csv and models/trained_models.mat\n\n');
end

%% ==================== List of functions ====================
function params = setParam(params, name, range)
% setParam turns on one hyperparameter and sets its search range

% disable default parameters if all are currently enabled
if all([params.Optimize])
    for i = 1:numel(params)
        params(i).Optimize = false;
    end
end

% enable requested parameter
for i = 1:numel(params)
    if strcmp(params(i).Name, name)
        params(i).Optimize = true;
        if ~isempty(range)
            params(i).Range = range;
        end
    end
end
end