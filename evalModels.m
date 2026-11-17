function R = evalModels(cfg)

if nargin < 1 || isempty(cfg), cfg = struct(); end
cfg = fillDefaults(cfg);
rng(cfg.seed, 'twister');
mkdirIf(cfg.outDir); mkdirIf(cfg.tabDir); mkdirIf(cfg.figDir);

specs       = cfg.models;
nM          = numel(specs);
modelNames  = string({specs.Name})';
metricNames = cfg.metricNames;
nMet        = numel(metricNames);
if nM < 2
    error('evalModels:needTwoModels', ...
        'At least two models are needed for the McNemar comparison.');
end

%% 1. Load data -----------------------------------------------------------
T = readtable(cfg.dataFile, 'TextType', 'string');
predNames = cfg.numVars(:)';
if cfg.useGenre, predNames = [predNames, {'genre'}]; end
absent = setdiff([predNames, {'viral'}], T.Properties.VariableNames);
if ~isempty(absent)
    error('evalModels:missingColumns', 'Columns not found in %s: %s', ...
        cfg.dataFile, strjoin(absent, ', '));
end
if cfg.useGenre, T.genre = categorical(T.genre); end
y = toBinary(T.viral);
P = T(:, predNames);          % predictors only: ids and `streams` are dropped
fprintf('Loaded %d rows, %d predictors. Viral rate = %.4f\n', ...
    height(T), numel(predNames), mean(y));

%% 2. Stratified hold-out split (test set used once) ----------------------
hp  = cvpartition(y, 'HoldOut', cfg.testFrac);
iTr = find(training(hp));
iTe = find(test(hp));
PTr = P(iTr, :);  yTr = y(iTr);
PTe = P(iTe, :);  yTe = y(iTe);
fprintf('Train n = %d (viral %.4f) | Test n = %d (viral %.4f)\n', ...
    numel(iTr), mean(yTr), numel(iTe), mean(yTe));

%% 3. Stratified k-fold CV on the training set ----------------------------
K   = cfg.kFolds;
cvp = cvpartition(yTr, 'KFold', K);     % same folds for every model
cvM = nan(K, nMet, nM);
for m = 1:nM
    for f = 1:K
        a = training(cvp, f);  b = test(cvp, f);
        prep = prepFit(PTr(a, :), cfg);                 % fit on fold-train only
        Xa = prepApply(PTr(a, :), prep);
        Xb = prepApply(PTr(b, :), prep);
        mdl = specs(m).Fit(Xa, yTr(a));
        [yh, sc] = scoreModel(mdl, Xb);
        cvM(f, :, m) = calcMetrics(yTr(b), yh, sc);
    end
    fprintf('CV done: %-22s %s = %.4f\n', specs(m).Name, cfg.selectMetric, ...
        mean(cvM(:, strcmp(metricNames, cfg.selectMetric), m)));
end

cvMean = nan(nM, nMet);  cvSD = nan(nM, nMet);
for m = 1:nM
    cvMean(m, :) = mean(cvM(:, :, m), 1);
    cvSD(m, :)   = std(cvM(:, :, m), 0, 1);
end

% Table: mean +/- SD (display strings) and numeric versions
cvStr = table(modelNames, 'VariableNames', {'Model'});
for j = 1:nMet
    col = strings(nM, 1);
    for m = 1:nM
        col(m) = sprintf('%.4f %s %.4f', cvMean(m, j), char(177), cvSD(m, j));
    end
    cvStr.(metricNames{j}) = col;
end
cvNum = [table(modelNames, 'VariableNames', {'Model'}), ...
         array2table(cvMean, 'VariableNames', strcat(metricNames, '_mean')), ...
         array2table(cvSD,   'VariableNames', strcat(metricNames, '_sd'))];

% Long table of per-fold results
foldModel = repelem(modelNames, K);
foldIdx   = repmat((1:K)', nM, 1);
foldVals  = zeros(nM * K, nMet);
for m = 1:nM, foldVals((m-1)*K + (1:K), :) = cvM(:, :, m); end
cvFolds = [table(foldModel, foldIdx, 'VariableNames', {'Model', 'Fold'}), ...
           array2table(foldVals, 'VariableNames', metricNames)];

%% 4. Model selection (CV only, test set not involved) --------------------
selCol = find(strcmp(metricNames, cfg.selectMetric), 1);
[~, rk] = sort(cvMean(:, selCol), 'descend');
iBest = rk(1);  iRun = rk(2);
selTbl = table(modelNames(rk), (1:nM)', cvMean(rk, selCol), cvSD(rk, selCol), ...
    'VariableNames', {'Model', 'Rank', ['CV_' cfg.selectMetric '_mean'], ...
                      ['CV_' cfg.selectMetric '_sd']});
fprintf('Best: %s | Runner-up: %s\n', modelNames(iBest), modelNames(iRun));

%% 5. Single test-set run --------------------------------------------------
prepF = prepFit(PTr, cfg);
Xtr   = prepApply(PTr, prepF);
Xte   = prepApply(PTe, prepF);
models = cell(nM, 1);
yhat   = zeros(numel(yTe), nM);
score  = zeros(numel(yTe), nM);
testM  = nan(nM, nMet);
for m = 1:nM
    models{m} = specs(m).Fit(Xtr, yTr);
    [yhat(:, m), score(:, m)] = scoreModel(models{m}, Xte);
    testM(m, :) = calcMetrics(yTe, yhat(:, m), score(:, m));
end
testTbl = [table(modelNames, 'VariableNames', {'Model'}), ...
           array2table(testM, 'VariableNames', metricNames)];
baseRate = table(mean(yTr), mean(yTe), 'VariableNames', ...
    {'ViralRate_train', 'ViralRate_test'});

%% 6. McNemar's test: best vs. runner-up ----------------------------------
mc = mcnemarTest(yTe, yhat(:, iBest), yhat(:, iRun), cfg.alpha);
mcTbl = table(modelNames(iBest), modelNames(iRun), mc.n11, mc.n10, mc.n01, ...
    mc.n00, mc.stat, mc.p, string(mc.method), mc.p < cfg.alpha, ...
    'VariableNames', {'ModelA_best', 'ModelB_runnerUp', 'BothCorrect', ...
    'OnlyA_correct', 'OnlyB_correct', 'BothWrong', 'Statistic', 'pValue', ...
    'Method', 'SignificantAtAlpha'});

%% 7. Permutation importance (best model, test set) -----------------------
bestMdl = models{iBest};
[~, auc0] = rocCurve(yTe, score(:, iBest));
f10 = calcMetrics(yTe, yhat(:, iBest), score(:, iBest));  f10 = f10(strcmp(metricNames, 'F1'));
nF  = numel(predNames);
dAUC = nan(cfg.nPermRepeats, nF);  dF1 = nan(cfg.nPermRepeats, nF);
for j = 1:nF
    for r = 1:cfg.nPermRepeats
        Pp = PTe;
        Pp.(predNames{j}) = Pp.(predNames{j})(randperm(height(Pp)));
        [yh, sc] = scoreModel(bestMdl, prepApply(Pp, prepF));
        mm = calcMetrics(yTe, yh, sc);
        dAUC(r, j) = auc0 - mm(strcmp(metricNames, 'ROC_AUC'));
        dF1(r, j)  = f10  - mm(strcmp(metricNames, 'F1'));
    end
end
impTbl = table(string(predNames'), mean(dAUC, 1)', std(dAUC, 0, 1)', ...
    mean(dF1, 1)', std(dF1, 0, 1)', 'VariableNames', ...
    {'Feature', 'AUCdrop_mean', 'AUCdrop_sd', 'F1drop_mean', 'F1drop_sd'});
impTbl = sortrows(impTbl, 'AUCdrop_mean', 'descend');

%% 8. Error analysis (best model, test set) -------------------------------
sB  = score(:, iBest);  yh = yhat(:, iBest);
isTP = yh == 1 & yTe == 1;  isTN = yh == 0 & yTe == 0;
isFP = yh == 1 & yTe == 0;  isFN = yh == 0 & yTe == 1;
isErr = isFP | isFN;

errGenre = table();
if cfg.useGenre
    g = categorical(PTe.genre);  cats = categories(g);
    nG = numel(cats);
    n = zeros(nG, 1); vr = n; er = n; fpr = n; fnr = n;
    for i = 1:nG
        ix = g == cats{i};
        n(i)   = sum(ix);
        vr(i)  = mean(yTe(ix));
        er(i)  = mean(isErr(ix));
        fpr(i) = sum(isFP(ix)) / max(sum(yTe(ix) == 0), 1);
        fnr(i) = sum(isFN(ix)) / max(sum(yTe(ix) == 1), 1);
    end
    errGenre = table(string(cats), n, vr, er, fpr, fnr, 'VariableNames', ...
        {'Genre', 'N', 'ViralRate', 'ErrorRate', 'FalsePositiveRate', 'FalseNegativeRate'});
    errGenre = sortrows(errGenre, 'ErrorRate', 'descend');
end

errYear = table();
if ismember('release_year', predNames)
    yr = PTe.release_year;
    edges = [min(yr), 2015, 2020, max(yr) + 1];
    edges = unique(edges);
    bin = discretize(yr, edges);
    nb = numel(edges) - 1;
    lab = strings(nb, 1); n = zeros(nb, 1); er = n; vr = n;
    for i = 1:nb
        lab(i) = sprintf('%d-%d', edges(i), edges(i+1) - 1);
        ix = bin == i;
        n(i) = sum(ix); er(i) = mean(isErr(ix)); vr(i) = mean(yTe(ix));
    end
    errYear = table(lab, n, vr, er, 'VariableNames', ...
        {'ReleaseYears', 'N', 'ViralRate', 'ErrorRate'});
end

% Standardized mean differences: errors vs. correctly classified, same true class
numNames = cfg.numVars(:)';
nN = numel(numNames);
smdFN = nan(nN, 1);  smdFP = nan(nN, 1);
for j = 1:nN
    x = double(PTe.(numNames{j}));  s = std(x);
    smdFN(j) = (mean(x(isFN)) - mean(x(isTP))) / s;   % missed viral vs caught viral
    smdFP(j) = (mean(x(isFP)) - mean(x(isTN))) / s;   % false alarms vs true non-viral
end
errShift = table(string(numNames'), smdFN, smdFP, 'VariableNames', ...
    {'Feature', 'SMD_FalseNeg_vs_TruePos', 'SMD_FalsePos_vs_TrueNeg'});

border = sB(isErr);
errSummary = table(sum(isFP), sum(isFN), mean(isErr), mean(sB(isErr)), ...
    mean(sB(~isErr)), 'VariableNames', {'FalsePositives', 'FalseNegatives', ...
    'ErrorRate', 'MeanScore_errors', 'MeanScore_correct'}); %#ok<NASGU>

errCases = table(iTe(isErr), yTe(isErr), yh(isErr), sB(isErr), ...
    'VariableNames', {'RowInCSV', 'Actual', 'Predicted', 'Score'});
if ismember('track_id', T.Properties.VariableNames)
    errCases.track_id = T.track_id(iTe(isErr));
end
if ismember('track_name', T.Properties.VariableNames)
    errCases.track_name = T.track_name(iTe(isErr));
end
if cfg.useGenre, errCases.genre = string(PTe.genre(isErr)); end
if ismember('release_year', predNames), errCases.release_year = PTe.release_year(isErr); end

%% 9. Write tables ---------------------------------------------------------
wt = @(tb, nm) writetable(tb, fullfile(cfg.tabDir, nm), 'Encoding', 'UTF-8');
wt(cvStr,   'table1_cv_mean_sd.csv');
wt(cvNum,   'table1b_cv_mean_sd_numeric.csv');
wt(cvFolds, 'table1c_cv_per_fold.csv');
wt(selTbl,  'table2_model_selection.csv');
wt(testTbl, 'table3_test_results.csv');
wt(baseRate, 'table3b_baseline_viral_rate.csv');
wt(mcTbl,   'table4_mcnemar_best_vs_runnerup.csv');
wt(impTbl,  'table5_permutation_importance.csv');
if ~isempty(errGenre), wt(errGenre, 'table6a_error_by_genre.csv'); end
if ~isempty(errYear),  wt(errYear,  'table6b_error_by_year.csv');  end
wt(errShift, 'table6c_error_feature_shift.csv');
wt(errCases, 'table6d_misclassified_test_tracks.csv');

%% 10. Figures (vector PDF) -------------------------------------------------
cols = lines(nM);

% Fig 1: CV metric mean +/- SD
fig = newFig([6.5 4.2]);  ax = axes(fig);  hold(ax, 'on');
off = linspace(-0.08, 0.08, K)';
for m = 1:nM
    scatter(ax, m + off, cvM(:, selCol, m), 18, [0.6 0.6 0.6], 'filled');
end
errorbar(ax, 1:nM, cvMean(:, selCol), cvSD(:, selCol), 'ko', ...
    'MarkerFaceColor', 'k', 'LineWidth', 1.2, 'CapSize', 8);
set(ax, 'XTick', 1:nM, 'XTickLabel', cellstr(modelNames), ...
    'XTickLabelRotation', 20, 'TickLabelInterpreter', 'none', 'FontSize', 10, 'Box', 'on');
ylabel(ax, sprintf('%s (CV, %d-fold)', strrep(cfg.selectMetric, '_', ' '), K));
title(ax, sprintf('Cross-validated %s: mean %s SD (grey = folds)', ...
    strrep(cfg.selectMetric, '_', ' '), char(177)));
grid(ax, 'on');  xlim(ax, [0.5 nM + 0.5]);
exportFig(fig, cfg, 'fig1_cv_metric_mean_sd');

% Fig 2: ROC curves (test)
fig = newFig([5.5 4.8]);  ax = axes(fig);  hold(ax, 'on');
for m = 1:nM
    [pts, auc] = rocCurve(yTe, score(:, m));
    lw = 1.3 + 0.9 * (m == iBest);
    plot(ax, pts(:, 1), pts(:, 2), 'Color', cols(m, :), 'LineWidth', lw, ...
        'DisplayName', sprintf('%s (AUC = %.3f)', modelNames(m), auc));
end
plot(ax, [0 1], [0 1], 'k--', 'DisplayName', 'Chance');
axis(ax, 'square');  xlim(ax, [0 1]);  ylim(ax, [0 1]);  grid(ax, 'on');
xlabel(ax, 'False positive rate');  ylabel(ax, 'True positive rate');
title(ax, 'ROC curves (test set)');
legend(ax, 'Location', 'southeast', 'Interpreter', 'none', 'FontSize', 8);
set(ax, 'FontSize', 10, 'Box', 'on');
exportFig(fig, cfg, 'fig2_roc_curves');

% Fig 3: Precision-recall curves (test)
fig = newFig([5.5 4.8]);  ax = axes(fig);  hold(ax, 'on');
for m = 1:nM
    [rec, prc, ap] = prCurve(yTe, score(:, m));
    lw = 1.3 + 0.9 * (m == iBest);
    plot(ax, rec, prc, 'Color', cols(m, :), 'LineWidth', lw, ...
        'DisplayName', sprintf('%s (AP = %.3f)', modelNames(m), ap));
end
yline(ax, mean(yTe), 'k--', 'DisplayName', sprintf('Baseline (%.3f)', mean(yTe)));
axis(ax, 'square');  xlim(ax, [0 1]);  ylim(ax, [0 1]);  grid(ax, 'on');
xlabel(ax, 'Recall');  ylabel(ax, 'Precision');
title(ax, 'Precision-recall curves (test set)');
legend(ax, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 8);
set(ax, 'FontSize', 10, 'Box', 'on');
exportFig(fig, cfg, 'fig3_pr_curves');

% Fig 4: Confusion matrix of the best model
tp = sum(isTP);  tn = sum(isTN);  fp = sum(isFP);  fn = sum(isFN);
C  = [tn fp; fn tp];
Cn = C ./ sum(C, 2);
fig = newFig([4.6 4.2]);  ax = axes(fig);
imagesc(ax, Cn);
colormap(ax, [linspace(1, 0.10, 256)', linspace(1, 0.35, 256)', linspace(1, 0.70, 256)']);
set(ax, 'CLim', [0 1]);
for i = 1:2
    for j = 1:2
        tc = 'k';  if Cn(i, j) > 0.5, tc = 'w'; end
        text(ax, j, i, sprintf('%d\n(%.1f%%)', C(i, j), 100 * Cn(i, j)), ...
            'HorizontalAlignment', 'center', 'Color', tc, 'FontSize', 11);
    end
end
set(ax, 'XTick', 1:2, 'XTickLabel', {'Non-viral', 'Viral'}, ...
    'YTick', 1:2, 'YTickLabel', {'Non-viral', 'Viral'}, 'FontSize', 10);
xlabel(ax, 'Predicted');  ylabel(ax, 'Actual');  axis(ax, 'equal', 'tight');
title(ax, sprintf('%s: confusion matrix (test)', modelNames(iBest)), 'Interpreter', 'none');
exportFig(fig, cfg, 'fig4_confusion_matrix_best');

% Fig 5: Permutation importance
nShow = min(nF, 15);
top = impTbl(1:nShow, :);
fig = newFig([6 4.6]);  ax = axes(fig);  hold(ax, 'on');
yy = (nShow:-1:1)';
barh(ax, yy, top.AUCdrop_mean, 0.6, 'FaceColor', [0.2 0.4 0.7]);
errorbar(ax, top.AUCdrop_mean, yy, [], [], top.AUCdrop_sd, top.AUCdrop_sd, ...
    'k.', 'CapSize', 5);
set(ax, 'YTick', 1:nShow, 'YTickLabel', flipud(cellstr(top.Feature)), ...
    'TickLabelInterpreter', 'none', 'FontSize', 10, 'Box', 'on');
xlabel(ax, 'Drop in ROC AUC when feature is permuted');
title(ax, sprintf('Permutation importance: %s (test, %d repeats)', ...
    modelNames(iBest), cfg.nPermRepeats), 'Interpreter', 'none');
grid(ax, 'on');
exportFig(fig, cfg, 'fig5_permutation_importance');

% Fig 6: Error analysis
fig = newFig([9 4]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
if ~isempty(errGenre)
    ax1 = nexttile(tl);
    bar(ax1, 100 * [errGenre.FalsePositiveRate, errGenre.FalseNegativeRate]);
    set(ax1, 'XTick', 1:height(errGenre), 'XTickLabel', cellstr(errGenre.Genre), ...
        'XTickLabelRotation', 45, 'TickLabelInterpreter', 'none', 'FontSize', 9);
    ylabel(ax1, 'Rate (%)');  grid(ax1, 'on');
    legend(ax1, {'False-positive rate', 'False-negative rate'}, 'FontSize', 8);
    title(ax1, 'Errors by genre');
end
ax2 = nexttile(tl);  hold(ax2, 'on');
edges = linspace(min(sB), max(sB), 26);
histogram(ax2, sB(~isErr), 'BinEdges', edges, 'Normalization', 'probability', ...
    'FaceAlpha', 0.6);
histogram(ax2, sB(isErr), 'BinEdges', edges, 'Normalization', 'probability', ...
    'FaceAlpha', 0.6);
legend(ax2, {'Correct', 'Errors'}, 'FontSize', 8);
xlabel(ax2, 'Model score for "viral"');  ylabel(ax2, 'Share of group');
title(ax2, 'Score distribution: correct vs. errors');  grid(ax2, 'on');
set(ax2, 'FontSize', 9);
exportFig(fig, cfg, 'fig6_error_analysis');

%% 11. Save + console summary ------------------------------------------------
R = struct('cfg', cfg, 'cvMean', cvMean, 'cvSD', cvSD, 'cvFolds', cvFolds, ...
    'cvTable', cvStr, 'selection', selTbl, 'test', testTbl, 'mcnemar', mc, ...
    'importance', impTbl, 'errGenre', errGenre, 'errYear', errYear, ...
    'errShift', errShift, 'errCases', errCases, 'bestModel', modelNames(iBest), ...
    'runnerUp', modelNames(iRun), 'models', {models}, 'trainIdx', iTr, 'testIdx', iTe);
save(fullfile(cfg.outDir, 'evalResults.mat'), 'R');

fid = fopen(fullfile(cfg.outDir, 'results_summary.txt'), 'w');
cleaner = onCleanup(@() fclose(fid));
pr = @(varargin) fprintf(fid, varargin{:});
pr('Evaluation summary\n==================\n');
pr('Data: %s | n = %d | viral rate = %.4f\n', cfg.dataFile, height(T), mean(y));
pr('Split: train %d / test %d (stratified, seed %d)\n', numel(iTr), numel(iTe), cfg.seed);
pr('CV: %d-fold stratified on training set, selection metric = %s\n\n', K, cfg.selectMetric);
pr('Best model: %s | Runner-up: %s\n', modelNames(iBest), modelNames(iRun));
pr('McNemar (%s): statistic = %.4f, p = %.4g -> %s at alpha = %.2f\n', mc.method, ...
    mc.stat, mc.p, ternary(mc.p < cfg.alpha, 'significant', 'not significant'), cfg.alpha);
pr('Top permutation-importance features: %s\n', strjoin(cellstr(impTbl.Feature(1:min(3, nF))), ', '));
pr('Best-model test errors: %d FP, %d FN (error rate %.4f)\n', fp, fn, mean(isErr));
fprintf('\nDone. Tables: %s | Figures: %s\n', cfg.tabDir, cfg.figDir);
disp(cvStr); disp(testTbl); disp(mcTbl);
end

%% ======================= local functions =================================

function cfg = fillDefaults(cfg)
def.dataFile = 'viral_songs.csv';
def.outDir = 'results';
def.seed = 42;
def.testFrac = 0.20;        % plan lists both 70/15/15 and 80/20; CV replaces the validation set
def.kFolds = 5;
def.nPermRepeats = 20;
def.selectMetric = 'F1';    % one of metricNames
def.alpha = 0.05;
def.numVars = {'release_year', 'duration_sec', 'danceability', 'energy', ...
    'valence', 'tempo_bpm', 'acousticness', 'instrumentalness', ...
    'speechiness', 'liveness', 'loudness_db'};
def.logVars = {'acousticness', 'instrumentalness', 'speechiness'};
def.useGenre = true;
def.models = defaultModels();
def.metricNames = {'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC'};
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}) || isempty(cfg.(fn{i})), cfg.(fn{i}) = def.(fn{i}); end
end
cfg.tabDir = fullfile(cfg.outDir, 'tables');
cfg.figDir = fullfile(cfg.outDir, 'figures');
end

function specs = defaultModels()
% Fit handles take (X, y) with y numeric 0/1 and return a trained classifier.
% Replace with tuned hyperparameters from the tuning step via cfg.models.
specs = struct('Name', {}, 'Fit', {});
specs(end+1) = struct('Name', 'Logistic Regression', 'Fit', ...
    @(X, y) fitclinear(X, y, 'Learner', 'logistic', 'Regularization', 'ridge', 'Solver', 'lbfgs'));
specs(end+1) = struct('Name', 'Naive Bayes', 'Fit', @(X, y) fitcnb(X, y));
specs(end+1) = struct('Name', 'Decision Tree', 'Fit', ...
    @(X, y) fitctree(X, y, 'MinLeafSize', 20));
specs(end+1) = struct('Name', 'Random Forest', 'Fit', ...
    @(X, y) fitcensemble(X, y, 'Method', 'Bag', 'NumLearningCycles', 200, ...
    'Learners', templateTree('NumVariablesToSample', max(1, round(sqrt(size(X, 2)))), ...
    'MinLeafSize', 5)));
specs(end+1) = struct('Name', 'Gradient Boosted Trees', 'Fit', ...
    @(X, y) fitcensemble(X, y, 'Method', 'LogitBoost', 'NumLearningCycles', 300, ...
    'LearnRate', 0.1, 'Learners', templateTree('MaxNumSplits', 8)));
end

function y = toBinary(v)
if islogical(v) || isnumeric(v)
    if any(isnan(v)), error('evalModels:badTarget', 'Target `viral` has missing values.'); end
    y = double(v ~= 0);
else
    s = lower(strtrim(string(v)));
    y = double(ismember(s, ["1", "true", "yes", "y", "viral"]));
end
y = y(:);
end

% ---------- preprocessing (fit on training data only) ----------------------
function prep = prepFit(P, cfg)
num = cfg.numVars(:)';
Xn = numMatrix(P, num);
prep.num = num;
prep.med = median(Xn, 1, 'omitnan');        % guard: data has no missing values
Xn = fillNaN(Xn, prep.med);
prep.lo = prctile(Xn, 1, 1);                % winsorize at train 1st/99th percentile
prep.hi = prctile(Xn, 99, 1);
Xn = min(max(Xn, prep.lo), prep.hi);
prep.logIdx = ismember(num, cfg.logVars);   % log1p on skewed features
Xn(:, prep.logIdx) = log1p(max(Xn(:, prep.logIdx), 0));
prep.mu = mean(Xn, 1);
prep.sd = std(Xn, 0, 1);
prep.sd(prep.sd == 0) = 1;
prep.useGenre = cfg.useGenre;
if cfg.useGenre
    gs = unique(string(P.genre));
    prep.genres = sort(gs(~ismissing(gs)));
    prep.genres = prep.genres(:);
end
end

function [X, names] = prepApply(P, prep)
Xn = numMatrix(P, prep.num);
Xn = fillNaN(Xn, prep.med);
Xn = min(max(Xn, prep.lo), prep.hi);
Xn(:, prep.logIdx) = log1p(max(Xn(:, prep.logIdx), 0));
X = (Xn - prep.mu) ./ prep.sd;
names = prep.num;
if prep.useGenre
    G = double(string(P.genre) == prep.genres');   % one-hot, unseen genre -> all zeros
    X = [X, G];
    names = [names, cellstr("genre_" + prep.genres')];
end
end

function Xn = numMatrix(P, num)
% Column-by-column so integer columns (e.g. release_year) never force
% the whole matrix into an integer class.
Xn = zeros(height(P), numel(num));
for j = 1:numel(num), Xn(:, j) = double(P.(num{j})); end
end

function X = fillNaN(X, med)
for j = 1:size(X, 2)
    m = isnan(X(:, j));
    if any(m), X(m, j) = med(j); end
end
end

% ---------- scoring and metrics --------------------------------------------
function [yh, sc] = scoreModel(mdl, X)
[lab, s] = predict(mdl, X);
col = find(mdl.ClassNames == 1, 1);
sc = s(:, col);
yh = double(lab(:) == 1);
end

function m = calcMetrics(y, yh, sc)
tp = sum(y == 1 & yh == 1);  fp = sum(y == 0 & yh == 1);
fn = sum(y == 1 & yh == 0);  tn = sum(y == 0 & yh == 0);
acc  = (tp + tn) / numel(y);
prec = tp / max(tp + fp, 1);
rec  = tp / max(tp + fn, 1);
f1   = 0;  if prec + rec > 0, f1 = 2 * prec * rec / (prec + rec); end
[~, auc] = rocCurve(y, sc);
[~, ~, ap] = prCurve(y, sc);
m = [acc, prec, rec, f1, auc, ap];
end

function [pts, auc] = rocCurve(y, sc)
[fpr, tpr, ~, auc] = perfcurve(y, sc, 1);
pts = [fpr, tpr];
end

function [rec, prc, ap] = prCurve(y, sc)
% Precision-recall curve and average precision (step-wise area), tie-aware.
[ss, ord] = sort(sc(:), 'descend');
pos = y(ord) == 1;
tp = cumsum(pos);  fp = cumsum(~pos);
last = [diff(ss) ~= 0; true];
tp = tp(last);  fp = fp(last);
prc = tp ./ (tp + fp);
rec = tp / sum(pos);
ap  = sum(diff([0; rec]) .* prc);
rec = [0; rec];  prc = [prc(1); prc];
end

% ---------- McNemar's test ---------------------------------------------------
function mc = mcnemarTest(y, yA, yB, alpha) %#ok<INUSD>
cA = yA(:) == y(:);  cB = yB(:) == y(:);
mc.n11 = sum(cA & cB);   mc.n10 = sum(cA & ~cB);
mc.n01 = sum(~cA & cB);  mc.n00 = sum(~cA & ~cB);
d = mc.n10 + mc.n01;
if d == 0
    mc.stat = 0;  mc.p = 1;  mc.method = 'no disagreements';
elseif d < 25                                   % exact binomial for small counts
    mc.stat = min(mc.n10, mc.n01);
    mc.p = min(1, 2 * binocdf(min(mc.n10, mc.n01), d, 0.5));
    mc.method = 'exact binomial';
else                                            % chi-square with continuity correction
    mc.stat = (abs(mc.n10 - mc.n01) - 1)^2 / d;
    mc.p = chi2cdf(mc.stat, 1, 'upper');
    mc.method = 'chi-square, continuity-corrected';
end
end

% ---------- small helpers ----------------------------------------------------
function fig = newFig(sizeIn)
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 sizeIn]);
end

function exportFig(fig, cfg, name)
f = fullfile(cfg.figDir, [name '.pdf']);
try
    exportgraphics(fig, f, 'ContentType', 'vector', 'BackgroundColor', 'white');
catch
    pos = fig.Position;
    set(fig, 'PaperUnits', 'inches', 'PaperSize', pos(3:4), 'PaperPosition', [0 0 pos(3:4)]);
    print(fig, f, '-dpdf', '-painters');
end
end

function mkdirIf(d)
if ~exist(d, 'dir'), mkdir(d); end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
