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
SEED = 42;
rng(SEED, 'twister'); % Fixed random seed for full end-to-end reproducibility
tPipeline = tic;

%% 1. Data Preprocessing & Leakage Verification (Groupmate 1 - Untouched)
fprintf('=== Step 1: Running Leakage & Preprocessing Self-Checks ===\n');
testPrepData("viral_songs.csv");

fprintf('\n=== Step 2: Running Leakage-Free Preprocessing ===\n');
D = prepData("viral_songs.csv", 'Seed', SEED);

%% 2. Exploratory Data Analysis Figures & Data Dictionary (Groupmate 1 - Untouched)
fprintf('\n=== Step 3: Generating EDA Figures & Data Dictionary ===\n');
edaFigures("viral_songs.csv", "figures");

%% 3. Bayesian Model Tuning, Hyperparameter Logging & Saving (Your Code)
fprintf('\n=== Step 4: Tuning Models, Logging Hyperparameters & Saving ===\n');
[cfg, tunedMdls, HyperparamTable] = trainModels(D, SEED); %#ok<ASGLU>

%% 4. Evaluation Harness: CV, Test Run, ROC/PR, McNemar & Error Analysis (Groupmate 2 - Untouched)
fprintf('=== Step 5: Running Evaluation Harness (evalModels.m) ===\n');
R = evalModels(cfg);

fprintf('\nEnd-to-end pipeline complete in %.1f seconds (Seed = %d). Best Model: %s\n', ...
    toc(tPipeline), SEED, R.bestModel);