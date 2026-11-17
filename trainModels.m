function out = trainModels(D, options)
%TRAINMODELS  Fast, lecture-aligned training of 6 models across 5 families
%             using 'OptimizeHyperparameters' and a shared cvpartition object.

arguments
    D = []
    options.Seed (1,1) double = 42
    options.MaxEvals (1,1) double {mustBeInteger, mustBePositive} = 10
    options.ModelsDir (1,1) string = "models"
    options.ResultsDir (1,1) string = "results"
    options.IncludeGenre (1,1) logical = true
    options.Verbose (1,1) logical = true
end

tStart = tic;
rng(options.Seed);

if isempty(D)
    D = prepData("viral_songs.csv", 'Seed', options.Seed, ...
        'IncludeGenre', options.IncludeGenre, 'Verbose', false);
end
if ~isfolder(options.ModelsDir),  mkdir(options.ModelsDir);  end
if ~isfolder(options.ResultsDir), mkdir(options.ResultsDir); end

Xtrain = D.Xtrain{:,:};
ytrain = double(D.ytrain);
cv     = D.cv;
featNames = string(D.featureNames(:));

hpOpts = struct( ...
    'CVPartition',             cv, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', options.MaxEvals, ...
    'ShowPlots',               false, ...
    'Verbose',                 0);

modelNames = {'LogisticRegression', 'NaiveBayes', 'DecisionTree', ...
              'RandomForest', 'GradientBoostedTrees', 'NeuralNetwork'};
families   = {'Linear (Baseline)', 'Probabilistic', 'Tree', ...
              'Ensemble (Bagging)', 'Ensemble (Boosting)', 'Neural Network'};
slugs      = {'logistic_regression', 'naive_bayes', 'decision_tree', ...
              'random_forest', 'gradient_boosted_trees', 'neural_network'};

Models = struct();
cvRows = cell(6, 1);

if options.Verbose
    fprintf('\n=== Training 6 Models (5-Fold CV OptimizeHyperparameters, Seed=%d) ===\n', options.Seed);
    drawnow;
end

for i = 1:6
    name = modelNames{i};
    rng(options.Seed);
    tMdl = tic;

    switch name
        case 'LogisticRegression'
            mdl = fitclinear(Xtrain, ytrain, 'Learner', 'logistic', ...
                'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
                'HyperparameterOptimizationOptions', hpOpts);

        case 'NaiveBayes'
            pNB = hyperparameters('fitcnb', Xtrain, ytrain);
            for j = 1:numel(pNB)
                switch pNB(j).Name
                    case 'DistributionNames', pNB(j).Optimize = true;
                    case 'Width',             pNB(j).Range = [0.30, 1.50]; pNB(j).Optimize = true;
                    otherwise,                pNB(j).Optimize = false;
                end
            end
            hpNB = hpOpts;
            hpNB.MaxObjectiveEvaluations = 2;
            mdl = fitcnb(Xtrain, ytrain, ...
                'OptimizeHyperparameters', pNB, ...
                'HyperparameterOptimizationOptions', hpNB);

        case 'DecisionTree'
            mdl = fitctree(Xtrain, ytrain, ...
                'OptimizeHyperparameters', {'MaxNumSplits', 'MinLeafSize'}, ...
                'HyperparameterOptimizationOptions', hpOpts);

        case 'RandomForest'
            pRF = hyperparameters('fitcensemble', Xtrain, ytrain, 'Tree');
            for j = 1:numel(pRF)
                switch pRF(j).Name
                    case 'NumLearningCycles',    pRF(j).Range = [20, 60]; pRF(j).Optimize = true;
                    case 'MinLeafSize',          pRF(j).Range = [10, 40]; pRF(j).Optimize = true;
                    case 'NumVariablesToSample', pRF(j).Range = [3, min(10, size(Xtrain, 2))]; pRF(j).Optimize = true;
                    otherwise,                   pRF(j).Optimize = false;
                end
            end
            mdl = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', ...
                'OptimizeHyperparameters', pRF, ...
                'HyperparameterOptimizationOptions', hpOpts);

        case 'GradientBoostedTrees'
            pGB = hyperparameters('fitcensemble', Xtrain, ytrain, 'Tree');
            for j = 1:numel(pGB)
                switch pGB(j).Name
                    case 'NumLearningCycles', pGB(j).Range = [20, 60];     pGB(j).Optimize = true;
                    case 'LearnRate',         pGB(j).Range = [0.03, 0.25]; pGB(j).Optimize = true;
                    case 'MaxNumSplits',      pGB(j).Range = [2, 12];      pGB(j).Optimize = true;
                    otherwise,                pGB(j).Optimize = false;
                end
            end
            mdl = fitcensemble(Xtrain, ytrain, 'Method', 'LogitBoost', ...
                'OptimizeHyperparameters', pGB, ...
                'HyperparameterOptimizationOptions', hpOpts);

        case 'NeuralNetwork'
            pNN = hyperparameters('fitcnet', Xtrain, ytrain);
            for j = 1:numel(pNN)
                switch pNN(j).Name
                    case 'LayerSizes',  pNN(j).Range = [8, 64];      pNN(j).Optimize = true;
                    case 'Activations', pNN(j).Optimize = true;
                    case 'Lambda',      pNN(j).Range = [1e-5, 1e-1]; pNN(j).Optimize = true;
                    otherwise,          pNN(j).Optimize = false;
                end
            end
            mdl = fitcnet(Xtrain, ytrain, 'Standardize', false, 'IterationLimit', 120, ...
                'OptimizeHyperparameters', pNN, ...
                'HyperparameterOptimizationOptions', hpOpts);
    end

    dt = toc(tMdl);
    Models.(name) = mdl;

    cvMdl = crossval(mdl, 'CVPartition', cv);
    [oofPred, oofScore] = kfoldPredict(cvMdl);
    foldAcc = 1 - kfoldLoss(cvMdl, 'Mode', 'individual');

    K = cv.NumTestSets;
    f1 = zeros(K,1); pr = zeros(K,1); roc = zeros(K,1); prec = zeros(K,1); rec = zeros(K,1);
    for k = 1:K
        idx = test(cv, k);
        yk = ytrain(idx); pk = oofPred(idx); sk = oofScore(idx, 2);
        C = confusionmat(yk, pk, 'Order', [0, 1]);
        prec(k) = C(2,2) / max(C(2,2) + C(1,2), 1);
        rec(k)  = C(2,2) / max(C(2,2) + C(2,1), 1);
        f1(k)   = (2 * prec(k) * rec(k)) / max(prec(k) + rec(k), eps);
        [~, ~, ~, roc(k)] = perfcurve(yk, sk, 1);
        [~, ~, ~, pr(k)]  = perfcurve(yk, sk, 1, 'XCrit', 'reca', 'YCrit', 'prec');
    end

    if options.Verbose
        fprintf('[%d/6] %-21s | CV F1: %.4f +/- %.4f | PR-AUC: %.4f | ROC-AUC: %.4f | %.1fs\n', ...
            i, name, mean(f1), std(f1), mean(pr), mean(roc), dt);
        drawnow;
    end

    save(fullfile(options.ModelsDir, slugs{i} + ".mat"), 'mdl', 'name', 'D', 'featNames');

    cvRows{i} = table(string(name), string(families{i}), mean(f1), std(f1), ...
        mean(pr), std(pr), mean(roc), std(roc), mean(prec), mean(rec), mean(foldAcc), dt, ...
        'VariableNames', {'Model','Family','CV_F1_Mean','CV_F1_SD','CV_PRAUC_Mean','CV_PRAUC_SD', ...
                          'CV_ROCAUC_Mean','CV_ROCAUC_SD','CV_Precision','CV_Recall','CV_Accuracy','TrainTimeSec'});
end

CVSummary = vertcat(cvRows{:});
writetable(CVSummary, fullfile(options.ResultsDir, 'cv_metrics_summary.csv'));

bundle = struct('Models', Models, 'PreprocessParams', D.Params, ...
    'CVSummary', CVSummary, 'FeatureNames', featNames, ...
    'Seed', options.Seed, 'TotalElapsedSec', toc(tStart));
save(fullfile(options.ModelsDir, 'trained_models.mat'), '-struct', 'bundle');
out = bundle;
end