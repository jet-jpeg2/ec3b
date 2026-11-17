function out = prepData(source, options)
%PREPDATA  Leakage-safe preprocessing for the "What Makes a Song Go Viral" dataset.
%
%   Every model in the project calls this one function, so all models see
%   identical preprocessing. Statistics (medians, caps, mean, std, genre list)
%   are learned from TRAINING rows only and then applied to any other rows.
%
%   With no input, prepData uses viral_songs.csv from the current folder, so the
%   Run button works as long as the CSV is in the same folder.
%
%   THREE WAYS TO CALL IT
%   1) Full pipeline (use this in main.m):
%        D = prepData("viral_songs.csv");
%      Imports, cleans, makes a stratified train/test split, fits all
%      parameters on the training rows, and returns
%        D.Xtrain, D.Xtest      tables of numeric predictors (z-scored + one-hot)
%        D.ytrain, D.ytest      0/1 viral labels (double column vectors)
%        D.cv                   stratified K-fold cvpartition on the TRAINING set
%        D.Params               fitted parameters (struct)
%        D.Report               per-feature EDA/preprocessing table
%        D.TrainTable/TestTable cleaned but unscaled rows (for per-fold refits)
%        D.featureNames, D.Info
%
%   2) Fit only (use this inside each CV fold so the validation fold never
%      influences scaling; this is the strictest form of leakage prevention):
%        F = prepData(TtrainFold, 'Split', false);   % F.X, F.y, F.Params
%
%   3) Apply saved parameters to new rows (validation fold or test set):
%        V = prepData(TvalFold, 'Params', F.Params); % V.X, V.y
%
%   Per-fold example:
%        for k = 1:D.cv.NumTestSets
%            Ttr = D.TrainTable(training(D.cv,k),:);
%            Tva = D.TrainTable(test(D.cv,k),:);
%            F = prepData(Ttr, 'Split', false);
%            V = prepData(Tva, 'Params', F.Params);
%            % train on F.X/F.y, score on V.X/V.y
%        end
%
%   NAME-VALUE OPTIONS
%     Seed                42     random seed for the split and the folds
%     TestFraction        0.20   hold-out fraction (stratified on viral)
%     NumFolds            5      folds for D.cv (training set only)
%     Split               true   false = fit on all rows given (no hold-out)
%     Params              []     fitted parameters; non-empty = apply only
%     IncludeGenre        true   false = drop genre (the "no genre" variant)
%     DropYear            false  true = drop release_year
%     ZeroFlags           false  true = add *_isZero flags for zero-inflated features
%     DropReferenceGenre  false  true = 9 genre dummies instead of 10
%                                (use for fitglm to avoid collinearity with the intercept)
%     CapPercentiles      [1 99] winsorizing percentiles (from training rows)
%     Verbose             true   print a summary after the full pipeline
%
%   STEPS (in order, matching the preprocessing plan)
%     1 Import       readtable, genre -> categorical
%     2 Drop         track_id, track_name, artist (identifiers) and streams
%                    (streams is a post-outcome variable that leaks the target)
%     3 Clean        remove duplicate track_id, trim genre labels
%     4 Split        stratified hold-out on viral, fixed seed (full pipeline only)
%     5 Missing      training median (numeric) / training mode (genre)
%     6 Outliers     cap at training 1st/99th percentiles (no rows removed)
%     7 Engineer     log1p on acousticness, instrumentalness, speechiness
%     8 Scale        z-score using training mean and std
%     9 Encode       one-hot genre (category list from training rows)
%
%   Requires: Statistics and Machine Learning Toolbox, MATLAB R2019b or later.

arguments
    source = "viral_songs.csv"
    options.Seed (1,1) double = 42
    options.TestFraction (1,1) double = 0.20
    options.NumFolds (1,1) double = 5
    options.Split (1,1) logical = true
    options.Params = []
    options.IncludeGenre (1,1) logical = true
    options.DropYear (1,1) logical = false
    options.ZeroFlags (1,1) logical = false
    options.DropReferenceGenre (1,1) logical = false
    options.CapPercentiles (1,2) double = [1 99]
    options.Verbose (1,1) logical = true
end

assert(options.TestFraction > 0 && options.TestFraction < 1, ...
    'prepData:badFraction', 'TestFraction must be between 0 and 1.');
assert(options.NumFolds >= 2 && options.NumFolds == round(options.NumFolds), ...
    'prepData:badFolds', 'NumFolds must be an integer >= 2.');

applyOnly = ~isempty(options.Params);
doSplit   = options.Split && ~applyOnly;

% ---- Steps 1-3: import, drop, clean (row-level operations, no statistics) ----
if istable(source)
    T = source;
else
    T = importRaw(char(source));
end
[T, clean] = cleanTable(T);

% ---- Apply-only mode: transform with saved parameters and return ----
if applyOnly
    P = options.Params;
    out.X = transformTable(T, P);
    if ismember('viral', T.Properties.VariableNames)
        out.y = T.viral;
    end
    out.Params = P;
    out.featureNames = P.featureNames;
    return
end

% ---- Step 4: stratified split (full pipeline) ----
if doSplit
    rng(options.Seed);
    cvOuter = cvpartition(T.viral, 'HoldOut', options.TestFraction);  % stratified on viral
    Ttr = T(training(cvOuter), :);
    Tte = T(test(cvOuter), :);
else
    Ttr = T;
end

% ---- Steps 5-9: learn every parameter from the training rows only ----
[P, report] = fitParams(Ttr, options);

% ---- Return ----
out.Params = P;
out.Report = report;
out.featureNames = P.featureNames;

if doSplit
    out.Xtrain = transformTable(Ttr, P);
    out.Xtest  = transformTable(Tte, P);
    out.ytrain = Ttr.viral;
    out.ytest  = Tte.viral;
    out.TrainTable = Ttr;
    out.TestTable  = Tte;
    out.cvOuter = cvOuter;
    rng(options.Seed);
    out.cv = cvpartition(out.ytrain, 'KFold', options.NumFolds);     % stratified on viral
    out.Info = struct( ...
        'Seed', options.Seed, ...
        'NTotal', height(T), ...
        'NTrain', height(Ttr), ...
        'NTest', height(Tte), ...
        'ViralRateAll', mean(T.viral), ...
        'ViralRateTrain', mean(Ttr.viral), ...
        'ViralRateTest', mean(Tte.viral), ...
        'DuplicatesRemoved', clean.duplicatesRemoved, ...
        'NumGenres', clean.numGenres, ...
        'NumFeatures', numel(P.featureNames));
    if options.Verbose
        printSummary(out.Info, report, options);
    end
else
    out.X = transformTable(Ttr, P);
    out.y = Ttr.viral;
end
end


%% ======================================================================
%  Local functions
%  ======================================================================

function lists = featureLists()
% Single place that defines which columns are used and how.
lists.numeric = {'release_year','duration_sec','danceability','energy','valence', ...
                 'tempo_bpm','acousticness','instrumentalness','speechiness', ...
                 'liveness','loudness_db'};
lists.logVars  = {'acousticness','instrumentalness','speechiness'};   % right-skewed, >= 0
lists.flagVars = {'acousticness','instrumentalness','speechiness','liveness'};  % zero-inflated
lists.dropVars = {'track_id','track_name','artist','streams'};
end

function T = importRaw(file)
% Step 1: import with genre as categorical.
opts = detectImportOptions(file);
opts = setvartype(opts, 'genre', 'categorical');
T = readtable(file, opts);
end

function [T, clean] = cleanTable(T)
% Steps 2-3: drop identifier / leaking columns, remove duplicates, tidy genre.
vn = T.Properties.VariableNames;
clean.rowsIn = height(T);

if ismember('track_id', vn)
    [~, keep] = unique(string(T.track_id), 'stable');   % duplicate track_id
    T = T(keep, :);
end
clean.duplicatesRemoved = clean.rowsIn - height(T);

T = removevars(T, intersect(featureLists().dropVars, T.Properties.VariableNames));

lists = featureLists();
need = [lists.numeric, {'genre'}];
absent = setdiff(need, T.Properties.VariableNames);
if ~isempty(absent)
    error('prepData:missingColumns', 'Missing required column(s): %s', strjoin(absent, ', '));
end

T.genre = categorical(strtrim(string(T.genre)));
if ismember('viral', T.Properties.VariableNames)
    T.viral = double(T.viral);
end
clean.numGenres = numel(categories(T.genre));
if clean.numGenres ~= 10
    warning('prepData:genreCount', 'Expected 10 genre labels but found %d.', clean.numGenres);
end
end

function [P, report] = fitParams(Ttr, o)
% Steps 5-9: every number below comes from the training rows only.
L = featureLists();
numVars = L.numeric;
if o.DropYear
    numVars = setdiff(numVars, {'release_year'}, 'stable');
end
nV = numel(numVars);

P.numVars   = numVars;
P.capVars   = ~strcmp(numVars, 'release_year');   % release_year is not capped
P.logVars   = ismember(numVars, L.logVars);
P.median    = zeros(1, nV);
P.capLo     = -inf(1, nV);
P.capHi     =  inf(1, nV);
P.flagVars  = {};
if o.ZeroFlags
    P.flagVars = L.flagVars;
end
P.includeGenre = o.IncludeGenre;

% Step 5 (median) and step 6 (percentile caps)
raw = zeros(height(Ttr), nV);
for j = 1:nV
    x = double(Ttr.(numVars{j}));
    P.median(j) = median(x, 'omitnan');
    x(isnan(x)) = P.median(j);
    raw(:, j) = x;
    if P.capVars(j)
        pc = prctile(x, o.CapPercentiles);
        P.capLo(j) = pc(1);
        P.capHi(j) = pc(2);
    end
end

% Genre categories and mode (strings, so unseen/undefined labels are safe)
g = string(Ttr.genre);
g = g(~ismissing(g));
[cats, ~, ic] = unique(g);
[~, m] = max(accumarray(ic, 1));
P.genreMode = cats(m);
if o.DropReferenceGenre
    P.genreCols = cats(2:end);
else
    P.genreCols = cats;
end
if ~o.IncludeGenre
    P.genreCols = strings(0, 1);
end

% Feature names, in the column order of transformTable
flagNames  = cellfun(@(v) [v '_isZero'], P.flagVars, 'UniformOutput', false);
if isempty(P.genreCols)
    genreNames = {};
else
    genreNames = cellstr(matlab.lang.makeValidName("genre_" + P.genreCols));
end
P.featureNames = [P.numVars(:); flagNames(:); genreNames(:)]';

% Steps 7 and 8: log1p, then z-score parameters from the training rows
P.mu = zeros(1, nV);
P.sigma = ones(1, nV);
E = engineer(Ttr, P);
P.mu = mean(E, 1);
P.sigma = std(E, 0, 1);
P.sigma(P.sigma == 0) = 1;

% Per-feature report: the evidence behind each preprocessing step
tf = repmat("z-score", nV, 1);
for j = 1:nV
    if P.capVars(j) && P.logVars(j)
        tf(j) = "cap + log1p + z-score";
    elseif P.capVars(j)
        tf(j) = "cap + z-score";
    end
end
skewRaw   = skewness(raw, 0, 1);
skewAfter = skewness(E, 0, 1);
outPct = zeros(nV, 1); zeroPct = zeros(nV, 1); cappedPct = zeros(nV, 1);
for j = 1:nV
    outPct(j)    = 100 * mean(isoutlier(raw(:, j), 'quartiles'));
    zeroPct(j)   = 100 * mean(raw(:, j) == 0);
    cappedPct(j) = 100 * mean(raw(:, j) < P.capLo(j) | raw(:, j) > P.capHi(j));
end
report = table(string(numVars(:)), skewRaw(:), skewAfter(:), outPct, zeroPct, ...
    P.capLo(:), P.capHi(:), cappedPct, tf, ...
    'VariableNames', {'Feature','SkewRaw','SkewAfter','IQROutlierPct','ZeroPct', ...
                      'CapLow','CapHigh','CappedPct','Transform'});
end

function [E, F] = engineer(T, P)
% Impute, cap, log1p (unscaled). F holds the optional zero-flags.
n = height(T);
nV = numel(P.numVars);
E = zeros(n, nV);
imputed = struct();
for j = 1:nV
    v = P.numVars{j};
    x = double(T.(v));
    x(isnan(x)) = P.median(j);
    imputed.(v) = x;                       % flags use the imputed, uncapped value
    if P.capVars(j)
        x = min(max(x, P.capLo(j)), P.capHi(j));
    end
    if P.logVars(j)
        x = log1p(max(x, 0));
    end
    E(:, j) = x;
end
F = zeros(n, numel(P.flagVars));
for k = 1:numel(P.flagVars)
    F(:, k) = double(imputed.(P.flagVars{k}) == 0);
end
end

function G = oneHot(T, P)
n = height(T);
if ~P.includeGenre
    G = zeros(n, 0);
    return
end
g = string(T.genre);
g(ismissing(g)) = P.genreMode;
G = zeros(n, numel(P.genreCols));
for k = 1:numel(P.genreCols)
    G(:, k) = double(g == P.genreCols(k));
end
end

function X = transformTable(T, P)
[E, F] = engineer(T, P);
Z = (E - P.mu) ./ P.sigma;
X = array2table([Z, F, oneHot(T, P)], 'VariableNames', P.featureNames);
end

function printSummary(info, report, o)
fprintf('\n=== prepData summary (seed %d) ===\n', info.Seed);
fprintf('Rows: %d total | %d train | %d test | duplicates removed: %d\n', ...
    info.NTotal, info.NTrain, info.NTest, info.DuplicatesRemoved);
fprintf('Viral rate: all %.2f%% | train %.2f%% | test %.2f%%\n', ...
    100*info.ViralRateAll, 100*info.ViralRateTrain, 100*info.ViralRateTest);
fprintf('Genres: %d | predictors after encoding: %d | IncludeGenre=%d DropYear=%d ZeroFlags=%d\n', ...
    info.NumGenres, info.NumFeatures, o.IncludeGenre, o.DropYear, o.ZeroFlags);
disp(report);
end