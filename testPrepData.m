function testPrepData(csvFile)
%TESTPREPDATA  Self-checks for prepData.m, including the leakage tests.
%   Run:  testPrepData("viral_songs.csv")
%   Every check prints PASS or stops with an error. Jet signs off on the
%   program only after this runs clean in a fresh MATLAB session.
arguments
    csvFile (1,1) string = "viral_songs.csv"
end

D = prepData(csvFile, 'Verbose', false);
tol = 1e-10;
n = 0;

% 1. Sizes: 80/20 split of 10,000 rows; 11 numeric + 10 genre = 21 predictors
check(D.Info.NTotal == 10000, 'dataset has 10,000 rows');
check(D.Info.NTrain == 8000 && D.Info.NTest == 2000, '80/20 split gives 8000 / 2000 rows');
check(numel(D.featureNames) == 21 && width(D.Xtrain) == 21, '21 predictors after encoding');
check(D.Info.DuplicatesRemoved == 0, 'no duplicate track_id');

% 2. Stratification: viral rate preserved in both parts
check(abs(D.Info.ViralRateTrain - D.Info.ViralRateAll) < 0.005, 'train viral rate matches full data (+/-0.5 pp)');
check(abs(D.Info.ViralRateTest  - D.Info.ViralRateAll) < 0.005, 'test viral rate matches full data (+/-0.5 pp)');
check(abs(mean(D.ytrain) - 0.3605) < 0.005, 'viral rate is about 36.05%');

% 3. Target-leakage columns never reach the predictors
bad = intersect(lower(D.featureNames), {'viral','streams','track_id','track_name','artist'});
check(isempty(bad), 'viral, streams and identifiers are not predictors');

% 4. Everything is finite and numeric
check(all(isfinite(D.Xtrain{:,:}), 'all') && all(isfinite(D.Xtest{:,:}), 'all'), 'no NaN or Inf in Xtrain / Xtest');

% 5. Scaling was learned from TRAIN: train z-scores are mean 0 / std 1,
%    test z-scores are close but NOT exactly 0 / 1
numCols = D.Params.numVars;
mTrain = mean(D.Xtrain{:, numCols}, 1);
sTrain = std(D.Xtrain{:, numCols}, 0, 1);
mTest  = mean(D.Xtest{:, numCols}, 1);
check(all(abs(mTrain) < 1e-9), 'train numeric columns have mean 0');
check(all(abs(sTrain - 1) < 1e-9), 'train numeric columns have std 1');
check(any(abs(mTest) > 1e-6), 'test columns are NOT re-centred (test statistics were not used)');

% 6. Capping uses training percentiles and also works on extreme new values
P = D.Params;
j = find(strcmp(P.numVars, 'duration_sec'));
Tx = D.TestTable(1:2, :);
Tx.duration_sec(1) = P.capHi(j);          % exactly at the cap
Tx.duration_sec(2) = 1e6;                 % absurd outlier
V = prepData(Tx, 'Params', P);
check(abs(V.X.duration_sec(1) - V.X.duration_sec(2)) < tol, 'extreme test value is capped to the training 99th percentile');

% 7. Leakage test: parameters depend ONLY on the training rows.
%    Refit from the training table alone and compare with the full pipeline.
F = prepData(D.TrainTable, 'Split', false);
check(max(abs(F.Params.mu - P.mu)) < tol && max(abs(F.Params.sigma - P.sigma)) < tol, ...
    'refit on training rows reproduces mean/std exactly');
check(max(abs(F.Params.capHi - P.capHi)) < tol && max(abs(F.Params.median - P.median)) < tol, ...
    'refit on training rows reproduces caps and medians exactly');
check(max(max(abs(F.X{:,:} - D.Xtrain{:,:}))) < tol, 'refit reproduces Xtrain exactly');

% 8. Applying saved parameters to the test table reproduces Xtest
V = prepData(D.TestTable, 'Params', P);
check(max(max(abs(V.X{:,:} - D.Xtest{:,:}))) < tol, 'apply mode reproduces Xtest exactly');

% 9. Absurd test values cannot break or shift the pipeline
Tcorrupt = D.TestTable;
Tcorrupt.energy(:) = 1e5;
Vc = prepData(Tcorrupt, 'Params', P);
check(all(isfinite(Vc.X{:,:}), 'all'), 'corrupted test data still gives finite output');
check(max(abs(Vc.X.energy - Vc.X.energy(1))) < tol, 'all corrupted energy values are capped to the same training-based value');

% 10. Reproducibility: same seed gives same split, other seed gives another
D2 = prepData(csvFile, 'Verbose', false);
D3 = prepData(csvFile, 'Seed', 7, 'Verbose', false);
check(isequal(training(D.cvOuter), training(D2.cvOuter)), 'same seed gives the identical split');
check(~isequal(training(D.cvOuter), training(D3.cvOuter)), 'different seed gives a different split');

% 11. Folds: stratified, cover the training set, and per-fold refit works
check(D.cv.NumTestSets == 5, '5-fold partition on the training set');
for k = 1:D.cv.NumTestSets
    Ttr = D.TrainTable(training(D.cv, k), :);
    Tva = D.TrainTable(test(D.cv, k), :);
    Fk = prepData(Ttr, 'Split', false);
    Vk = prepData(Tva, 'Params', Fk.Params);
    check(abs(mean(Tva.viral) - mean(D.ytrain)) < 0.01, sprintf('fold %d is stratified (+/-1 pp)', k));
    check(all(abs(mean(Fk.X{:, numCols}, 1)) < 1e-9) && any(abs(mean(Vk.X{:, numCols}, 1)) > 1e-6), ...
        sprintf('fold %d: validation fold scaled with its own training fold only', k));
end

% 12. Option variants used in the experiments
Dng = prepData(csvFile, 'IncludeGenre', false, 'Verbose', false);
check(width(Dng.Xtrain) == 11, 'no-genre variant has 11 predictors');
Dny = prepData(csvFile, 'DropYear', true, 'Verbose', false);
check(width(Dny.Xtrain) == 20 && ~ismember('release_year', Dny.featureNames), 'no-year variant has 20 predictors');
Dzf = prepData(csvFile, 'ZeroFlags', true, 'Verbose', false);
check(width(Dzf.Xtrain) == 25, 'zero-flag variant has 25 predictors');
Dref = prepData(csvFile, 'DropReferenceGenre', true, 'Verbose', false);
check(width(Dref.Xtrain) == 20, 'reference-genre variant has 20 predictors');

fprintf('\nAll %d checks passed.\n', n);

    function check(cond, msg)
        n = n + 1;
        if ~cond
            error('testPrepData:failed', 'FAIL: %s', msg);
        end
        fprintf('PASS  %s\n', msg);
    end
end