%% ========================================================================
%  LBYEC3B PROJECT: What Makes a Song Go Viral?
%  Predicting Track Virality from Spotify Audio Features and Genre
%
%  Submitted by: Dotollo, Melanie S.
%                Meneses, Jet Ryan S.
%                Vigo, Karol Miel F.
%  Submitted to: Dr. Edwin Sybingco
%  Course:       LBYEC3B - Intelligent Systems for Engineering (DLSU-Manila)
%% ========================================================================
clear; clc; close all;
rng(42); % Fixed random seed for reproducibility across all models
tPipeline = tic;

if ~isfolder("models"),  mkdir("models");  end
if ~isfolder("results"), mkdir("results"); end
if ~isfolder("figures"), mkdir("figures"); end

%% ========================================================================
%  TASK 1: DATA LOADING, CLEANING & LEAKAGE-FREE PREPROCESSING
%  ========================================================================
fprintf('--- Task 1: Data Loading & Leakage-Free Preprocessing ---\n');

raw = readtable("viral_songs.csv");

% 1.1 Deduplication using unique(..., 'stable') (Lab 3 & Lab 6)
[~, keepIdx] = unique(string(raw.track_id), 'stable');
data = raw(keepIdx, :);

% 1.2 Drop identifier columns and post-outcome 'streams' (prevents target leakage)
data = removevars(data, {'track_id', 'track_name', 'artist', 'streams'});

numCols = {'release_year', 'duration_sec', 'danceability', 'energy', 'valence', ...
           'tempo_bpm', 'acousticness', 'instrumentalness', 'speechiness', ...
           'liveness', 'loudness_db'};
logCols = {'acousticness', 'instrumentalness', 'speechiness'};

X_num_all = data{:, numCols};
genre_all = categorical(strtrim(string(data.genre)));
Y = double(data.viral);

% 1.3 Stratified Train/Test Split (80% Train, 20% Test) using cvpartition (Lab 4)
cvHoldout = cvpartition(Y, 'HoldOut', 0.20);
X_tr_raw = X_num_all(training(cvHoldout), :);
G_tr_raw = genre_all(training(cvHoldout));
Y_train  = Y(training(cvHoldout));

X_te_raw = X_num_all(test(cvHoldout), :);
G_te_raw = genre_all(test(cvHoldout));
Y_test   = Y(test(cvHoldout));

% 1.4 Missing Value Imputation Using Training Medians & Mode Only (Lab 3 & Lab 4)
train_medians = median(X_tr_raw, 'omitnan');
X_tr_imp = fillmissing(X_tr_raw, 'constant', train_medians);
X_te_imp = fillmissing(X_te_raw, 'constant', train_medians);

genre_mode = mode(G_tr_raw);
G_tr_imp = fillmissing(G_tr_raw, 'constant', genre_mode);
G_te_imp = fillmissing(G_te_raw, 'constant', genre_mode);

% 1.5 Outlier Capping (Winsorization) Using Training 1st/99th Percentiles (Lab 3-6)
%     Note: release_year (col 1) is bounded by domain years and not capped.
lowerBound = prctile(X_tr_imp, 1);
upperBound = prctile(X_tr_imp, 99);
lowerBound(1) = -inf;
upperBound(1) =  inf;

X_tr_cap = max(min(X_tr_imp, upperBound), lowerBound);
X_te_cap = max(min(X_te_imp, upperBound), lowerBound);

% 1.6 Log1p Transform for Right-Skewed Audio Features (acousticness, instrumentalness, speechiness)
logIdx = ismember(numCols, logCols);
X_tr_cap(:, logIdx) = log1p(max(X_tr_cap(:, logIdx), 0));
X_te_cap(:, logIdx) = log1p(max(X_te_cap(:, logIdx), 0));

% 1.7 Z-Score Standardization Without Leakage (Lab 4-6)
[X_tr_scaled, mu_train, sigma_train] = zscore(X_tr_cap);
X_te_scaled = (X_te_cap - mu_train) ./ sigma_train;

% 1.8 One-Hot Encoding for Genre using dummyvar(categorical(...)) (Lab 5)
genreCats = categories(G_tr_imp);
G_tr_dummy = dummyvar(categorical(G_tr_imp, genreCats));
G_te_dummy = dummyvar(categorical(G_te_imp, genreCats));

X_train = [X_tr_scaled, G_tr_dummy];
X_test  = [X_te_scaled, G_te_dummy];
featureNames = [numCols, cellstr("genre_" + string(genreCats))'];

% 1.9 Shared Stratified 5-Fold CV Partition on Training Set (Used by ALL Models)
rng(42);
cv = cvpartition(Y_train, 'KFold', 5);

fprintf('Preprocessing complete: %d train | %d test | %d features | Viral rate = %.2f%%\n\n', ...
    size(X_train, 1), size(X_test, 1), size(X_train, 2), 100 * mean(Y_train));
drawnow;


%% ========================================================================
%  TASK 2: MODEL TRAINING & HYPERPARAMETER TUNING (6 MODELS, 5 FAMILIES)
%  ========================================================================
fprintf('--- Task 2: Training & Bayesian Hyperparameter Optimization ---\n');
drawnow;

maxEvals = 10; % Bayesian optimization budget per model
hpOpts = struct( ...
    'CVPartition',             cv, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', maxEvals, ...
    'ShowPlots',               false, ...
    'Verbose',                 0);

modelNames = { ...
    'LogisticRegression (Baseline)', ...
    'NaiveBayes', ...
    'DecisionTree', ...
    'RandomForest (Bag)', ...
    'GradientBoostedTrees', ...
    'NeuralNetwork'};
modelFamilies = {'Linear (Baseline)', 'Probabilistic', 'Tree', ...
                 'Ensemble (Bagging)', 'Ensemble (Boosting)', 'Neural Network'};

models   = cell(6, 1);
trainSec = zeros(6, 1);

% 1. Logistic Regression (Interpretable Linear Baseline: fitclinear)
rng(42); t0 = tic;
models{1} = fitclinear(X_train, Y_train, ...
    'Learner', 'logistic', ...
    'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
    'HyperparameterOptimizationOptions', hpOpts);
trainSec(1) = toc(t0);
fprintf('[1/6] %-30s tuned in %.1fs\n', modelNames{1}, trainSec(1)); drawnow;

% 2. Naive Bayes (Probabilistic Family: fitcnb)
rng(42); t0 = tic;
paramsNB = hyperparameters('fitcnb', X_train, Y_train);
for j = 1:numel(paramsNB)
    switch paramsNB(j).Name
        case 'DistributionNames', paramsNB(j).Optimize = true;
        case 'Width',             paramsNB(j).Range = [0.30, 1.50]; paramsNB(j).Optimize = true;
        otherwise,                paramsNB(j).Optimize = false;
    end
end
hpOptsNB = hpOpts;
hpOptsNB.MaxObjectiveEvaluations = 2;
models{2} = fitcnb(X_train, Y_train, ...
    'OptimizeHyperparameters', paramsNB, ...
    'HyperparameterOptimizationOptions', hpOptsNB);
trainSec(2) = toc(t0);
fprintf('[2/6] %-30s tuned in %.1fs\n', modelNames{2}, trainSec(2)); drawnow;

% 3. Decision Tree (Tree Family: fitctree)
rng(42); t0 = tic;
models{3} = fitctree(X_train, Y_train, ...
    'OptimizeHyperparameters', {'MaxNumSplits', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', hpOpts);
trainSec(3) = toc(t0);
fprintf('[3/6] %-30s tuned in %.1fs\n', modelNames{3}, trainSec(3)); drawnow;

% 4. Random Forest (Bagged Trees Ensemble: fitcensemble 'Bag')
rng(42); t0 = tic;
paramsRF = hyperparameters('fitcensemble', X_train, Y_train, 'Tree');
for j = 1:numel(paramsRF)
    switch paramsRF(j).Name
        case 'NumLearningCycles',    paramsRF(j).Range = [20, 60]; paramsRF(j).Optimize = true;
        case 'MinLeafSize',          paramsRF(j).Range = [10, 40]; paramsRF(j).Optimize = true;
        case 'NumVariablesToSample', paramsRF(j).Range = [3, 10];  paramsRF(j).Optimize = true;
        otherwise,                   paramsRF(j).Optimize = false;
    end
end
models{4} = fitcensemble(X_train, Y_train, 'Method', 'Bag', ...
    'OptimizeHyperparameters', paramsRF, ...
    'HyperparameterOptimizationOptions', hpOpts);
trainSec(4) = toc(t0);
fprintf('[4/6] %-30s tuned in %.1fs\n', modelNames{4}, trainSec(4)); drawnow;

% 5. Gradient Boosted Trees (Boosting Ensemble: fitcensemble 'LogitBoost')
rng(42); t0 = tic;
paramsGB = hyperparameters('fitcensemble', X_train, Y_train, 'Tree');
for j = 1:numel(paramsGB)
    switch paramsGB(j).Name
        case 'NumLearningCycles', paramsGB(j).Range = [20, 60];     paramsGB(j).Optimize = true;
        case 'LearnRate',         paramsGB(j).Range = [0.03, 0.25]; paramsGB(j).Optimize = true;
        case 'MaxNumSplits',      paramsGB(j).Range = [2, 12];      paramsGB(j).Optimize = true;
        otherwise,                paramsGB(j).Optimize = false;
    end
end
models{5} = fitcensemble(X_train, Y_train, 'Method', 'LogitBoost', ...
    'OptimizeHyperparameters', paramsGB, ...
    'HyperparameterOptimizationOptions', hpOpts);
trainSec(5) = toc(t0);
fprintf('[5/6] %-30s tuned in %.1fs\n', modelNames{5}, trainSec(5)); drawnow;

% 6. Neural Network (Multilayer Perceptron: fitcnet)
rng(42); t0 = tic;
paramsNN = hyperparameters('fitcnet', X_train, Y_train);
for j = 1:numel(paramsNN)
    switch paramsNN(j).Name
        case 'LayerSizes',  paramsNN(j).Range = [8, 64];        paramsNN(j).Optimize = true;
        case 'Activations', paramsNN(j).Optimize = true;
        case 'Lambda',      paramsNN(j).Range = [1e-5, 1e-1];   paramsNN(j).Optimize = true;
        otherwise,          paramsNN(j).Optimize = false;
    end
end
models{6} = fitcnet(X_train, Y_train, ...
    'Standardize', false, ...
    'IterationLimit', 120, ...
    'OptimizeHyperparameters', paramsNN, ...
    'HyperparameterOptimizationOptions', hpOpts);
trainSec(6) = toc(t0);
fprintf('[6/6] %-30s tuned in %.1fs\n\n', modelNames{6}, trainSec(6)); drawnow;


%% ========================================================================
%  TASK 3: STRATIFIED 5-FOLD CV & HELD-OUT TEST EVALUATION
%  ========================================================================
fprintf('--- Task 3: 5-Fold Cross-Validation & Held-Out Test Evaluation ---\n');
drawnow;

cvF1Mean = zeros(6,1); cvF1SD = zeros(6,1);
cvPRMean = zeros(6,1); cvROCMean = zeros(6,1); cvAccMean = zeros(6,1);

testF1 = zeros(6,1); testPR = zeros(6,1); testROC = zeros(6,1);
testPrec = zeros(6,1); testRec = zeros(6,1); testAcc = zeros(6,1);
inferMs = zeros(6,1);

testPreds  = cell(6, 1);
testScores = cell(6, 1);
rocCurves  = cell(6, 1);
prCurves   = cell(6, 1);

for i = 1:6
    % 3.1 Stratified 5-Fold Cross-Validation using crossval + kfoldPredict
    if isa(models{i}, 'ClassificationLinear')
        % ClassificationLinear stores 'ridge (L2)' / 'lasso (L1)', whereas
        % fitclinear expects 'ridge' or 'lasso'.
        regType = 'ridge';
        if contains(lower(string(models{i}.Regularization)), 'lasso')
            regType = 'lasso';
        end
        cvMdl = fitclinear(X_train, Y_train, ...
            'Learner', 'logistic', ...
            'Regularization', regType, ...
            'Lambda', models{i}.Lambda, ...
            'CVPartition', cv);
    else
        cvMdl = crossval(models{i}, 'CVPartition', cv);
    end

    [oofPred, oofScore] = kfoldPredict(cvMdl);
    foldAcc = 1 - kfoldLoss(cvMdl, 'Mode', 'individual');

    f1Fold = zeros(5, 1); prFold = zeros(5, 1); rocFold = zeros(5, 1);
    for k = 1:5
        idxVal = test(cv, k);
        yk = Y_train(idxVal);
        pk = oofPred(idxVal);
        sk = oofScore(idxVal, 2);

        mFold = computeBinaryMetrics(yk, pk);
        f1Fold(k) = mFold.F1;
        [~, ~, ~, rocFold(k)] = perfcurve(yk, sk, 1);
        [~, ~, ~, prFold(k)]  = perfcurve(yk, sk, 1, 'XCrit', 'reca', 'YCrit', 'prec');
    end

    cvF1Mean(i)  = mean(f1Fold);
    cvF1SD(i)    = std(f1Fold);
    cvPRMean(i)  = mean(prFold);
    cvROCMean(i) = mean(rocFold);
    cvAccMean(i) = mean(foldAcc);

    % 3.2 Final Single Evaluation on Held-Out Test Set using predict
    tInf = tic;
    [yPred, yScore] = predict(models{i}, X_test);
    inferMs(i) = 1000 * toc(tInf);

    testPreds{i}  = yPred;
    testScores{i} = yScore(:, 2);

    mTest = computeBinaryMetrics(Y_test, yPred);
    testF1(i)   = mTest.F1;
    testPrec(i) = mTest.Precision;
    testRec(i)  = mTest.Recall;
    testAcc(i)  = 1 - loss(models{i}, X_test, Y_test);

    [fpr, tpr, ~, testROC(i)] = perfcurve(Y_test, yScore(:, 2), 1);
    [recC, preC, ~, testPR(i)] = perfcurve(Y_test, yScore(:, 2), 1, 'XCrit', 'reca', 'YCrit', 'prec');
    rocCurves{i} = [fpr, tpr];
    prCurves{i}  = [recC, preC];
end

% 3.3 Streamlined Results Table (Primary & Supporting Metrics Only)
cvF1Summary = compose("%.4f ± %.4f", cvF1Mean, cvF1SD);

ResultsTable = table( ...
    string(modelNames(:)), string(modelFamilies(:)), ...
    cvF1Summary, round(testF1, 4), ...
    round(testPrec, 4), round(testRec, 4), ...
    round(testROC, 4), round(100 * testAcc, 2), round(trainSec, 2), ...
    'VariableNames', {'Model', 'Family', ...
                      'CV_F1_Mean_SD', 'Test_F1', ...
                      'Test_Precision', 'Test_Recall', ...
                      'Test_ROCAUC', 'Test_AccuracyPct', 'TrainTimeSec'});

disp(ResultsTable);
writetable(ResultsTable, fullfile("results", "model_comparison.csv"));
save(fullfile("models", "trained_models.mat"), 'models', 'modelNames', 'ResultsTable', ...
    'train_medians', 'genre_mode', 'lowerBound', 'upperBound', 'mu_train', 'sigma_train', ...
    'genreCats', 'featureNames');


%% ========================================================================
%  TASK 4: STATISTICAL SIGNIFICANCE CHECK (McNEMAR'S TEST) & ERROR ANALYSIS
%  ========================================================================
fprintf('--- Task 4: Statistical Significance Check & Error Analysis ---\n');

% 4.1 Compare Top Two Models Using McNemar's Test (testcholdout)
%     Note: testcholdout performs McNemar's test; 'Test' selects the p-value
%     variant ('asymptotic', 'exact', or 'midp').
[~, rankIdx] = sort(cvF1Mean, 'descend');
bestIdx   = rankIdx(1);
secondIdx = rankIdx(2);

[hMcNemar, pMcNemar, e1, e2] = testcholdout( ...
    testPreds{bestIdx}, testPreds{secondIdx}, Y_test, 'Test', 'midp');

fprintf('Top 1 Model: %s (CV F1 = %.4f +/- %.4f | Test F1 = %.4f)\n', ...
    modelNames{bestIdx}, cvF1Mean(bestIdx), cvF1SD(bestIdx), testF1(bestIdx));
fprintf('Top 2 Model: %s (CV F1 = %.4f +/- %.4f | Test F1 = %.4f)\n', ...
    modelNames{secondIdx}, cvF1Mean(secondIdx), cvF1SD(secondIdx), testF1(secondIdx));
fprintf('McNemar Test (testcholdout, midp): h = %d, p = %.4g (Error rates: %.2f%% vs %.2f%%)\n\n', ...
    hMcNemar, pMcNemar, 100*e1, 100*e2);

% 4.2 Feature Importance for Best Model
bestMdl = models{bestIdx};
if ismember(bestIdx, [3, 4, 5]) % Tree / Bagged / Boosted Ensemble
    impScores = predictorImportance(bestMdl)';
elseif bestIdx == 1 % Logistic Regression
    impScores = abs(bestMdl.Beta(:));
else
    % Permutation importance across features on training set
    baseAcc = 1 - loss(bestMdl, X_train, Y_train);
    impScores = zeros(numel(featureNames), 1);
    for f = 1:numel(featureNames)
        Xp = X_train;
        Xp(:, f) = Xp(randperm(size(Xp, 1)), f);
        impScores(f) = max(0, baseAcc - (1 - loss(bestMdl, Xp, Y_train)));
    end
end
impPct = 100 * impScores / max(sum(impScores), eps);
ImpTable = sortrows(table(string(featureNames(:)), impPct, ...
    'VariableNames', {'Feature', 'ImportancePct'}), 'ImportancePct', 'descend');
disp('Top 10 Most Important Features (Best Model):');
disp(ImpTable(1:min(10, height(ImpTable)), :));
writetable(ImpTable, fullfile("results", "best_model_feature_importance.csv"));

% 4.3 Error Analysis Breakdown by Genre (Best Model)
bestPred = testPreds{bestIdx};
isFP = (Y_test == 0) & (bestPred == 1);
isFN = (Y_test == 1) & (bestPred == 0);
isOK = (Y_test == bestPred);

ErrorByGenre = table(string(genreCats), zeros(numel(genreCats),1), zeros(numel(genreCats),1), ...
    zeros(numel(genreCats),1), zeros(numel(genreCats),1), ...
    'VariableNames', {'Genre', 'TestTracks', 'AccuracyPct', 'FalsePositives', 'FalseNegatives'});
for g = 1:numel(genreCats)
    mG = (G_te_imp == genreCats{g});
    ErrorByGenre.TestTracks(g)     = sum(mG);
    ErrorByGenre.AccuracyPct(g)    = 100 * mean(isOK(mG));
    ErrorByGenre.FalsePositives(g) = sum(isFP(mG));
    ErrorByGenre.FalseNegatives(g) = sum(isFN(mG));
end
disp('Error Analysis by Genre on Held-Out Test Set (Best Model):');
disp(ErrorByGenre);
writetable(ErrorByGenre, fullfile("results", "error_analysis_by_genre.csv"));


%% ========================================================================
%  TASK 5: VISUALIZATION OF RESULTS (LAB 4, LAB 5 & LAB 6 STYLE)
%  ========================================================================
fprintf('--- Task 5: Generating Visualizations ---\n');
drawnow;

% Figure 1: Confusion Matrix Chart for Best Model (Lab 4 Task 5.1)
f1Fig = figure('Name', 'Best Model Confusion Matrix', 'Units', 'normalized', 'Position', [0.05 0.52 0.40 0.40]);
cm = confusionchart(Y_test, bestPred, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm.Title = sprintf('Confusion Matrix: %s (Test F1 = %.3f)', modelNames{bestIdx}, testF1(bestIdx));
exportgraphics(f1Fig, fullfile("figures", "fig1_best_confusion_matrix.png"), 'Resolution', 300);

% Figure 2: ROC Curves & Precision-Recall Curves Across All 6 Models
f2Fig = figure('Name', 'ROC and Precision-Recall Curves', 'Units', 'normalized', 'Position', [0.48 0.52 0.46 0.40]);
subplot(1, 2, 1); hold on;
for i = 1:6
    ls = '-'; if i == 1, ls = '--'; end
    plot(rocCurves{i}(:,1), rocCurves{i}(:,2), ls, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (AUC=%.3f)', modelNames{i}, testROC(i)));
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Test Set ROC Curves'); legend('Location', 'southeast', 'FontSize', 7.5);
grid on; hold off;

subplot(1, 2, 2); hold on;
for i = 1:6
    ls = '-'; if i == 1, ls = '--'; end
    plot(prCurves{i}(:,1), prCurves{i}(:,2), ls, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (PR-AUC=%.3f)', modelNames{i}, testPR(i)));
end
xlabel('Recall'); ylabel('Precision');
title('Test Set Precision-Recall Curves'); legend('Location', 'northeast', 'FontSize', 7.5);
grid on; hold off;
exportgraphics(f2Fig, fullfile("figures", "fig2_roc_pr_curves.png"), 'Resolution', 300);

% Figure 3: Feature Importance & Error Analysis (Lab 5 / Lab 6 Style)
f3Fig = figure('Name', 'Feature Importance and Error Analysis', 'Units', 'normalized', 'Position', [0.05 0.06 0.44 0.38]);
subplot(1, 2, 1);
top10 = flipud(ImpTable(1:min(10, height(ImpTable)), :));
barh(top10.ImportancePct);
set(gca, 'YTick', 1:height(top10), 'YTickLabel', strrep(top10.Feature, '_', ' '));
xlabel('Relative Importance (%)');
title(sprintf('Top 10 Features (%s)', modelNames{bestIdx}));
grid on;

subplot(1, 2, 2);
bar([ErrorByGenre.FalsePositives, ErrorByGenre.FalseNegatives], 'grouped');
set(gca, 'XTick', 1:height(ErrorByGenre), 'XTickLabel', ErrorByGenre.Genre, 'XTickLabelRotation', 35);
ylabel('Count on Test Set');
title('Error Breakdown by Genre');
legend({'False Positives', 'False Negatives'}, 'Location', 'northeast');
grid on;
exportgraphics(f3Fig, fullfile("figures", "fig3_importance_and_errors.png"), 'Resolution', 300);

% Figure 4: Exploratory PCA Scree Plot & 3D t-SNE Feature Space (Lab 6 Tasks 3 & 5)
[~, ~, ~, ~, explained] = pca(X_tr_scaled);
rng(42);
Z_tsne = tsne(X_test, 'NumDimensions', 3, 'Perplexity', 30);

f4Fig = figure('Name', 'PCA Scree and 3D t-SNE Embedding', 'Units', 'normalized', 'Position', [0.51 0.06 0.44 0.38]);
subplot(1, 2, 1);
pareto(explained);
xlabel('Principal Component'); ylabel('Variance Explained (%)');
title('Audio Features PCA Scree Plot');
grid on;

subplot(1, 2, 2);
scatter3(Z_tsne(Y_test==0,1), Z_tsne(Y_test==0,2), Z_tsne(Y_test==0,3), 18, [0.6 0.65 0.75], 'filled'); hold on;
scatter3(Z_tsne(Y_test==1,1), Z_tsne(Y_test==1,2), Z_tsne(Y_test==1,3), 24, [0.85 0.2 0.2], 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2'); zlabel('t-SNE 3');
title('3D t-SNE Test Set Embedding');
legend({'Non-Viral (0)', 'Viral (1)'}, 'Location', 'northeast');
grid on; hold off;
exportgraphics(f4Fig, fullfile("figures", "fig4_pca_and_tsne.png"), 'Resolution', 300);

fprintf('All tasks complete in %.1f seconds! Results saved to models/, results/, and figures/.\n', toc(tPipeline));


%% ========================================================================
%  HELPER FUNCTION: Precision, Recall & F1-Score via confusionmat
%  ========================================================================
function m = computeBinaryMetrics(yTrue, yPred)
C = confusionmat(yTrue, yPred, 'Order', [0, 1]);
TN = C(1,1); FP = C(1,2);
FN = C(2,1); TP = C(2,2);
m.Precision = TP / max(TP + FP, 1);
m.Recall    = TP / max(TP + FN, 1);
m.F1        = (2 * m.Precision * m.Recall) / max(m.Precision + m.Recall, eps);
m.Accuracy  = (TP + TN) / max(sum(C(:)), 1);
end