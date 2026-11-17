function [cfg, tunedMdls, HyperparamTable] = trainModels(D, seed)
% TRAINMODELS Tunes 1 baseline and 5 models across 5 families via Bayesian
%   optimization with 5-fold CV on the training set, logs final tuned
%   hyperparameters, saves trained models to models/, and returns cfg for
%   evalModels.m.

if nargin < 2 || isempty(seed), seed = 42; end
if nargin < 1 || isempty(D), D = prepData("viral_songs.csv", 'Seed', seed); end
if ~isfolder("models"), mkdir("models"); end

X_train = D.Xtrain{:, :};
Y_train = D.ytrain;
cv      = D.cv;

%% 1. Bayesian Optimization with 5-Fold CV on Training Set
fprintf('--- Bayesian Hyperparameter Optimization (6 Models, 5-Fold CV, Seed = %d) ---\n', seed);
drawnow;

hpOpts = struct( ...
    'CVPartition',             cv, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 10, ...
    'ShowPlots',               false, ...
    'Verbose',                 0);

modelNames = {'Logistic Regression', 'Naive Bayes', 'Decision Tree', ...
              'Random Forest', 'Gradient Boosted Trees', 'Neural Network'};
modelFamilies = {'Linear (Baseline)', 'Probabilistic', 'Tree', ...
                 'Ensemble (Bagging)', 'Ensemble (Boosting)', 'Neural Network'};

tunedMdls = cell(6, 1);
tuneSec   = zeros(6, 1);

% [1/6] Logistic Regression (Baseline - Linear Family)
rng(seed, 'twister'); t0 = tic;
tunedMdls{1} = fitclinear(X_train, Y_train, 'Learner', 'logistic', ...
    'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
    'HyperparameterOptimizationOptions', hpOpts);
tuneSec(1) = toc(t0);
fprintf('[1/6] %-24s tuned in %.1fs\n', modelNames{1}, tuneSec(1)); drawnow;

% [2/6] Naive Bayes (Probabilistic Family)
rng(seed, 'twister'); t0 = tic;
paramsNB = setSearchSpace(hyperparameters('fitcnb', X_train, Y_train), ...
    {'DistributionNames', []}, {'Width', [0.30, 1.50]});
hpOptsNB = hpOpts; hpOptsNB.MaxObjectiveEvaluations = 2;
tunedMdls{2} = fitcnb(X_train, Y_train, ...
    'OptimizeHyperparameters', paramsNB, ...
    'HyperparameterOptimizationOptions', hpOptsNB);
tuneSec(2) = toc(t0);
fprintf('[2/6] %-24s tuned in %.1fs\n', modelNames{2}, tuneSec(2)); drawnow;

% [3/6] Decision Tree (Tree Family)
rng(seed, 'twister'); t0 = tic;
tunedMdls{3} = fitctree(X_train, Y_train, ...
    'OptimizeHyperparameters', {'MaxNumSplits', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', hpOpts);
tuneSec(3) = toc(t0);
fprintf('[3/6] %-24s tuned in %.1fs\n', modelNames{3}, tuneSec(3)); drawnow;

% [4/6] Random Forest (Ensemble - Bagging)
rng(seed, 'twister'); t0 = tic;
paramsRF = setSearchSpace(hyperparameters('fitcensemble', X_train, Y_train, 'Tree'), ...
    {'NumLearningCycles', [20, 60]}, {'MinLeafSize', [10, 40]}, {'NumVariablesToSample', [3, 10]});
tunedMdls{4} = fitcensemble(X_train, Y_train, 'Method', 'Bag', ...
    'OptimizeHyperparameters', paramsRF, ...
    'HyperparameterOptimizationOptions', hpOpts);
tuneSec(4) = toc(t0);
fprintf('[4/6] %-24s tuned in %.1fs\n', modelNames{4}, tuneSec(4)); drawnow;

% [5/6] Gradient Boosted Trees (Ensemble - Boosting)
rng(seed, 'twister'); t0 = tic;
paramsGB = setSearchSpace(hyperparameters('fitcensemble', X_train, Y_train, 'Tree'), ...
    {'NumLearningCycles', [20, 60]}, {'LearnRate', [0.03, 0.25]}, {'MaxNumSplits', [2, 12]});
tunedMdls{5} = fitcensemble(X_train, Y_train, 'Method', 'LogitBoost', ...
    'OptimizeHyperparameters', paramsGB, ...
    'HyperparameterOptimizationOptions', hpOpts);
tuneSec(5) = toc(t0);
fprintf('[5/6] %-24s tuned in %.1fs\n', modelNames{5}, tuneSec(5)); drawnow;

% [6/6] Neural Network (Multilayer Perceptron)
rng(seed, 'twister'); t0 = tic;
paramsNN = setSearchSpace(hyperparameters('fitcnet', X_train, Y_train), ...
    {'Layer_1_Size', [8, 64]}, {'Activations', []}, {'Lambda', [1e-5, 1e-1]});
tunedMdls{6} = fitcnet(X_train, Y_train, 'Standardize', false, 'IterationLimit', 120, ...
    'OptimizeHyperparameters', paramsNN, ...
    'HyperparameterOptimizationOptions', hpOpts);
tuneSec(6) = toc(t0);
fprintf('[6/6] %-24s tuned in %.1fs\n\n', modelNames{6}, tuneSec(6)); drawnow;

%% 2. Extract & Log Final Tuned Hyperparameters
bestLambdaLR = tunedMdls{1}.Lambda;
bestRegLR    = 'ridge';
if contains(lower(string(tunedMdls{1}.Regularization)), 'lasso')
    bestRegLR = 'lasso';
end

bestNB = tunedMdls{2}.HyperparameterOptimizationResults.XAtMinObjective;
bestDT = tunedMdls{3}.HyperparameterOptimizationResults.XAtMinObjective;
bestRF = tunedMdls{4}.HyperparameterOptimizationResults.XAtMinObjective;
bestGB = tunedMdls{5}.HyperparameterOptimizationResults.XAtMinObjective;
bestNN = tunedMdls{6}.HyperparameterOptimizationResults.XAtMinObjective;

nnLayerSizes = tunedMdls{6}.LayerSizes;
nnAct        = char(tunedMdls{6}.Activations);
nnLambda     = bestNN.Lambda;

if string(bestNB.DistributionNames) == "kernel" && isfinite(bestNB.Width)
    nbSummary = sprintf("DistributionNames='kernel', Width=%.4f", bestNB.Width);
else
    nbSummary = "DistributionNames='normal'";
end

tunedParamsSummary = [ ...
    sprintf("Regularization='%s', Lambda=%.4g", bestRegLR, bestLambdaLR); ...
    nbSummary; ...
    sprintf("MaxNumSplits=%d, MinLeafSize=%d", bestDT.MaxNumSplits, bestDT.MinLeafSize); ...
    sprintf("NumLearningCycles=%d, MinLeafSize=%d, NumVariablesToSample=%d", ...
            bestRF.NumLearningCycles, bestRF.MinLeafSize, bestRF.NumVariablesToSample); ...
    sprintf("NumLearningCycles=%d, LearnRate=%.4f, MaxNumSplits=%d", ...
            bestGB.NumLearningCycles, bestGB.LearnRate, bestGB.MaxNumSplits); ...
    sprintf("LayerSizes=%s, Activations='%s', Lambda=%.4g", ...
            mat2str(nnLayerSizes), nnAct, nnLambda)];

HyperparamTable = table( ...
    string(modelNames(:)), string(modelFamilies(:)), ...
    tunedParamsSummary, round(tuneSec, 2), ...
    'VariableNames', {'Model', 'Family', 'FinalTunedHyperparameters', 'TuneTimeSec'});

fprintf('--- Final Tuned Hyperparameters Log ---\n');
disp(HyperparamTable);
writetable(HyperparamTable, fullfile("models", "tuned_hyperparameters.csv"));

%% 3. Build Model Specs (cfg.models) & Save Trained Models
specs = struct('Name', {}, 'Fit', {});

specs(1) = struct('Name', modelNames{1}, 'Fit', ...
    @(X, y) fitclinear(X, y, 'Learner', 'logistic', ...
    'Regularization', bestRegLR, 'Lambda', bestLambdaLR));

if string(bestNB.DistributionNames) == "kernel" && isfinite(bestNB.Width)
    nbWidth = bestNB.Width;
    specs(2) = struct('Name', modelNames{2}, 'Fit', ...
        @(X, y) fitcnb(X, y, 'DistributionNames', 'kernel', 'Width', nbWidth));
else
    specs(2) = struct('Name', modelNames{2}, 'Fit', ...
        @(X, y) fitcnb(X, y, 'DistributionNames', 'normal'));
end

specs(3) = struct('Name', modelNames{3}, 'Fit', ...
    @(X, y) fitctree(X, y, 'MaxNumSplits', bestDT.MaxNumSplits, ...
    'MinLeafSize', bestDT.MinLeafSize));

tRF = templateTree('MinLeafSize', bestRF.MinLeafSize, ...
                   'NumVariablesToSample', bestRF.NumVariablesToSample);
specs(4) = struct('Name', modelNames{4}, 'Fit', ...
    @(X, y) fitcensemble(X, y, 'Method', 'Bag', ...
    'NumLearningCycles', bestRF.NumLearningCycles, 'Learners', tRF));

tGB = templateTree('MaxNumSplits', bestGB.MaxNumSplits);
specs(5) = struct('Name', modelNames{5}, 'Fit', ...
    @(X, y) fitcensemble(X, y, 'Method', 'LogitBoost', ...
    'NumLearningCycles', bestGB.NumLearningCycles, ...
    'LearnRate', bestGB.LearnRate, 'Learners', tGB));

specs(6) = struct('Name', modelNames{6}, 'Fit', ...
    @(X, y) fitcnet(X, y, 'Standardize', false, 'IterationLimit', 120, ...
    'LayerSizes', nnLayerSizes, 'Activations', nnAct, ...
    'Lambda', nnLambda));

trainedModels = cell(6, 1);
for m = 1:6
    rng(seed, 'twister');
    trainedModels{m} = specs(m).Fit(X_train, Y_train);
end

prepParams = D.Params;
save(fullfile("models", "trained_models.mat"), 'trainedModels', 'tunedMdls', ...
    'HyperparamTable', 'modelNames', 'modelFamilies', 'prepParams');
fprintf('Saved trained models and hyperparameter log to models/.\n\n');

cfg = struct('dataFile', 'viral_songs.csv', 'outDir', 'results', ...
             'seed', seed, 'selectMetric', 'F1', 'models', specs);
end

%% Helper: Enable only selected hyperparameters and apply custom bounds
function params = setSearchSpace(params, varargin)
for j = 1:numel(params)
    params(j).Optimize = false;
end
for k = 1:numel(varargin)
    pName  = varargin{k}{1};
    pRange = varargin{k}{2};
    idx = find(strcmp({params.Name}, pName), 1);
    if ~isempty(idx)
        params(idx).Optimize = true;
        if ~isempty(pRange)
            params(idx).Range = pRange;
        end
    end
end
end
% === END OF FILE ===