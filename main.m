function results = main(options)
%MAIN  Lecture-aligned end-to-end pipeline for "What Makes a Song Go Viral?".
%
%   Implements the full evaluation protocol:
%     1. Leakage-safe preprocessing & shared cvpartition (prepData.m)
%     2. 6 models across 5 families tuned via OptimizeHyperparameters (trainModels.m)
%     3. Held-out test evaluation using predict, loss, confusionmat & perfcurve
%     4. Statistical significance test between top 2 models via testcholdout (McNemar)
%     5. Feature importance, genre error analysis & Lab 4/5/6 visualizations

arguments
    options.CsvFile (1,1) string = "viral_songs.csv"
    options.Seed (1,1) double = 42
    options.MaxEvals (1,1) double {mustBeInteger, mustBePositive} = 10
    options.RunSelfChecks (1,1) logical = true
    options.RunEDA (1,1) logical = false
    options.ShowPlots (1,1) logical = true
    options.ModelsDir (1,1) string = "models"
    options.ResultsDir (1,1) string = "results"
    options.FiguresDir (1,1) string = "figures"
end

tAll = tic;
rng(options.Seed);
if ~isfolder(options.ModelsDir),  mkdir(options.ModelsDir);  end
if ~isfolder(options.ResultsDir), mkdir(options.ResultsDir); end
if ~isfolder(options.FiguresDir), mkdir(options.FiguresDir); end

if options.RunSelfChecks && isfile("testPrepData.m")
    testPrepData(options.CsvFile);
end

if options.RunEDA && isfile("edaFigures.m")
    edaFigures(options.CsvFile, options.FiguresDir);
end

D = prepData(options.CsvFile, 'Seed', options.Seed, 'Verbose', true);
trainOut = trainModels(D, 'Seed', options.Seed, 'MaxEvals', options.MaxEvals, ...
    'ModelsDir', options.ModelsDir, 'ResultsDir', options.ResultsDir, 'Verbose', true);

Xtest = D.Xtest{:,:};
ytest = double(D.ytest(:));
modelNames = {'LogisticRegression', 'NaiveBayes', 'DecisionTree', ...
              'RandomForest', 'GradientBoostedTrees', 'NeuralNetwork'};

testRows = cell(6, 1);
predsAll = cell(6, 1);
rocAll   = cell(6, 1);
prAll    = cell(6, 1);

for i = 1:6
    name = modelNames{i};
    mdl  = trainOut.Models.(name);

    tInf = tic;
    [yPred, yScore] = predict(mdl, Xtest);
    inferMs = 1000 * toc(tInf);

    predsAll{i} = yPred;
    C = confusionmat(ytest, yPred, 'Order', [0, 1]);
    prec = C(2,2) / max(C(2,2) + C(1,2), 1);
    rec  = C(2,2) / max(C(2,2) + C(2,1), 1);
    f1   = (2 * prec * rec) / max(prec + rec, eps);
    acc  = 1 - loss(mdl, Xtest, ytest);

    [fpr, tpr, ~, rocAUC] = perfcurve(ytest, yScore(:, 2), 1);
    [reca, preca, ~, prAUC] = perfcurve(ytest, yScore(:, 2), 1, 'XCrit', 'reca', 'YCrit', 'prec');
    rocAll{i} = [fpr, tpr];
    prAll{i}  = [reca, preca];

    testRows{i} = table(string(name), f1, prAUC, rocAUC, prec, rec, acc, inferMs, ...
        'VariableNames', {'Model','Test_F1','Test_PRAUC','Test_ROCAUC', ...
                          'Test_Precision','Test_Recall','Test_Accuracy','TestInferMs'});
end

TestTable = vertcat(testRows{:});
TestTable.F1_vs_Baseline = TestTable.Test_F1 - TestTable.Test_F1(1);
FinalResults = join(trainOut.CVSummary, TestTable, 'Keys', 'Model');
writetable(FinalResults, fullfile(options.ResultsDir, 'final_results.csv'));

fprintf('\n=== Final Model Comparison (Baseline vs. All Models) ===\n');
disp(FinalResults);
drawnow;

% Statistical Significance Check: Top 2 Models via McNemar's Test (testcholdout)
[~, rankIdx] = sort(FinalResults.CV_F1_Mean, 'descend');
[hMc, pMc, e1, e2] = testcholdout(predsAll{rankIdx(1)}, predsAll{rankIdx(2)}, ytest, 'Test', 'mcnemar');
fprintf('McNemar Test (%s vs %s): h = %d, p = %.4g (Errors: %.2f%% vs %.2f%%)\n\n', ...
    modelNames{rankIdx(1)}, modelNames{rankIdx(2)}, hMc, pMc, 100*e1, 100*e2);

% Feature Importance for Best Model
bestIdx = rankIdx(1);
bestMdl = trainOut.Models.(modelNames{bestIdx});
if ismember(bestIdx, [3, 4, 5])
    imp = predictorImportance(bestMdl)';
elseif bestIdx == 1
    imp = abs(bestMdl.Beta(:));
else
    baseAcc = 1 - loss(bestMdl, D.Xtrain{:,:}, double(D.ytrain));
    imp = zeros(numel(D.featureNames), 1);
    for f = 1:numel(D.featureNames)
        Xp = D.Xtrain{:,:};
        Xp(:, f) = Xp(randperm(size(Xp, 1)), f);
        imp(f) = max(0, baseAcc - (1 - loss(bestMdl, Xp, double(D.ytrain))));
    end
end
ImpTable = sortrows(table(string(D.featureNames(:)), 100 * imp / max(sum(imp), eps), ...
    'VariableNames', {'Feature', 'ImportancePct'}), 'ImportancePct', 'descend');
disp('Top 10 Most Important Features (Best Model):');
disp(ImpTable(1:min(10, height(ImpTable)), :));
writetable(ImpTable, fullfile(options.ResultsDir, 'best_model_feature_importance.csv'));

% Error Analysis Breakdown by Genre (Best Model)
bestPred = predsAll{bestIdx};
isFP = (ytest == 0) & (bestPred == 1);
isFN = (ytest == 1) & (bestPred == 0);
isOK = (ytest == bestPred);

G_te_str = string(D.TestTable.genre);
G_te_str(ismissing(G_te_str)) = D.Params.genreMode;
genreCats = D.Params.genreCols;

ErrorByGenre = table(string(genreCats(:)), zeros(numel(genreCats),1), zeros(numel(genreCats),1), ...
    zeros(numel(genreCats),1), zeros(numel(genreCats),1), ...
    'VariableNames', {'Genre', 'TestTracks', 'AccuracyPct', 'FalsePositives', 'FalseNegatives'});
for g = 1:numel(genreCats)
    mG = (G_te_str == genreCats(g));
    ErrorByGenre.TestTracks(g)     = sum(mG);
    ErrorByGenre.AccuracyPct(g)    = 100 * mean(isOK(mG));
    ErrorByGenre.FalsePositives(g) = sum(isFP(mG));
    ErrorByGenre.FalseNegatives(g) = sum(isFN(mG));
end
disp('Error Analysis by Genre on Held-Out Test Set (Best Model):');
disp(ErrorByGenre);
writetable(ErrorByGenre, fullfile(options.ResultsDir, 'error_analysis_by_genre.csv'));
drawnow;

% Visualizations (Lab 4, Lab 5 & Lab 6 Style)
vis = 'off'; if options.ShowPlots, vis = 'on'; end

% Figure 1: Confusion Matrix Chart for Best Model
f1Fig = figure('Name', 'Best Model Confusion Matrix', 'Visible', vis, 'Units', 'normalized', 'Position', [0.05 0.52 0.40 0.40]);
cm = confusionchart(ytest, bestPred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('Confusion Matrix: %s (Test F1 = %.3f)', modelNames{bestIdx}, FinalResults.Test_F1(bestIdx));
exportgraphics(f1Fig, fullfile(options.FiguresDir, 'fig1_best_confusion_matrix.png'), 'Resolution', 300);

% Figure 2: ROC Curves & Precision-Recall Curves Across All 6 Models
f2Fig = figure('Name', 'ROC and PR Curves', 'Visible', vis, 'Units', 'normalized', 'Position', [0.48 0.52 0.46 0.40]);
subplot(1, 2, 1); hold on;
for i = 1:6
    ls = '-'; if i == 1, ls = '--'; end
    plot(rocAll{i}(:,1), rocAll{i}(:,2), ls, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (%.3f)', modelNames{i}, TestTable.Test_ROCAUC(i)));
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Test ROC Curves'); legend('Location', 'southeast', 'FontSize', 7.5); grid on; hold off;

subplot(1, 2, 2); hold on;
for i = 1:6
    ls = '-'; if i == 1, ls = '--'; end
    plot(prAll{i}(:,1), prAll{i}(:,2), ls, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (%.3f)', modelNames{i}, TestTable.Test_PRAUC(i)));
end
xlabel('Recall'); ylabel('Precision');
title('Test Precision-Recall Curves'); legend('Location', 'northeast', 'FontSize', 7.5); grid on; hold off;
exportgraphics(f2Fig, fullfile(options.FiguresDir, 'fig2_roc_pr_curves.png'), 'Resolution', 300);

% Figure 3: Feature Importance & Genre Error Breakdown
f3Fig = figure('Name', 'Feature Importance and Error Analysis', 'Visible', vis, 'Units', 'normalized', 'Position', [0.05 0.06 0.44 0.38]);
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
exportgraphics(f3Fig, fullfile(options.FiguresDir, 'fig3_importance_and_errors.png'), 'Resolution', 300);

% Figure 4: Exploratory PCA Scree Plot & 3D t-SNE Feature Space
numCols = D.Params.numVars;
[~, ~, ~, ~, explained] = pca(D.Xtrain{:, numCols});
rng(options.Seed);
Z_tsne = tsne(Xtest, 'NumDimensions', 3, 'Perplexity', 30);

f4Fig = figure('Name', 'PCA Scree and 3D t-SNE Embedding', 'Visible', vis, 'Units', 'normalized', 'Position', [0.51 0.06 0.44 0.38]);
subplot(1, 2, 1);
pareto(explained);
xlabel('Principal Component'); ylabel('Variance Explained (%)');
title('Audio Features PCA Scree Plot');
grid on;

subplot(1, 2, 2);
scatter3(Z_tsne(ytest==0,1), Z_tsne(ytest==0,2), Z_tsne(ytest==0,3), 18, [0.6 0.65 0.75], 'filled'); hold on;
scatter3(Z_tsne(ytest==1,1), Z_tsne(ytest==1,2), Z_tsne(ytest==1,3), 24, [0.85 0.2 0.2], 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2'); zlabel('t-SNE 3');
title('3D t-SNE Test Set Embedding');
legend({'Non-Viral (0)', 'Viral (1)'}, 'Location', 'northeast');
grid on; hold off;
exportgraphics(f4Fig, fullfile(options.FiguresDir, 'fig4_pca_and_tsne.png'), 'Resolution', 300);

fprintf('Pipeline complete in %.1f seconds.\n', toc(tAll));
results = struct('Data', D, 'Train', trainOut, 'FinalResults', FinalResults, ...
                 'FeatureImportance', ImpTable, 'ErrorByGenre', ErrorByGenre);
end