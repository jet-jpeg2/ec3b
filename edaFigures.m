function edaFigures(csvFile, outDir)
%EDAFIGURES  EDA figures and tables for the technical paper and slide deck.
%   Run:  edaFigures("viral_songs.csv", "figures")
%
%   Writes vector PDF (for the paper) and 300 dpi PNG (for the slides) to outDir:
%     fig1_eda_overview   7.16 in wide, 4 panels  -> paper Fig. 1 (two-column figure)
%     eda_class_balance   3.45 in (one column)
%     eda_genre_rate      3.45 in
%     eda_feature_dist    7.16 in, 10 features split by class
%     eda_correlation     7.16 in, correlation heatmap (annotated, readable in grayscale)
%     eda_skew_log1p      7.16 in, before / after log1p on the 3 skewed features
%     eda_year_rate       3.45 in, viral rate by release year
%     eda_streams_leak    3.45 in, why streams is excluded
%   and three CSV tables:
%     data_dictionary.csv       -> paper TABLE II
%     eda_feature_summary.csv   mean, SD, range, skew, zeros, IQR outliers, correlation with viral
%     eda_genre_summary.csv     tracks and viral rate per genre
%
%   Requires: Statistics and Machine Learning Toolbox, MATLAB R2020a or later
%   (boxchart, exportgraphics). Figure text is Times New Roman, 8 pt.

arguments
    csvFile (1,1) string = "viral_songs.csv"
    outDir (1,1) string = "figures"
end
if ~isfolder(outDir), mkdir(outDir); end

T = readtable(csvFile);
T.genre = categorical(strtrim(string(T.genre)));
T.viral = double(T.viral);

numVars = {'release_year','duration_sec','danceability','energy','valence', ...
           'tempo_bpm','acousticness','instrumentalness','speechiness', ...
           'liveness','loudness_db'};
col.viral = [0.10 0.25 0.50];     % dark: viral
col.non   = [0.78 0.78 0.78];     % light: non-viral (differs in luminance, so grayscale-safe)
W1 = 3.45; W2 = 7.16;             % IEEE single / double column width (inches)

%% ---- Console summary (numbers to quote in the paper) ----
N = height(T);
fprintf('\n=== EDA summary ===\n');
fprintf('Rows: %d | columns: %d | missing cells: %d\n', N, width(T), sum(ismissing(T), 'all'));
fprintf('Duplicate track_id: %d | genres: %d\n', N - numel(unique(T.track_id)), numel(categories(T.genre)));
fprintf('Viral: %d (%.2f%%) | non-viral: %d (%.2f%%)\n', sum(T.viral==1), 100*mean(T.viral), ...
    sum(T.viral==0), 100*(1-mean(T.viral)));

%% ---- Tables ----
writetable(buildDictionary(T), fullfile(outDir, 'data_dictionary.csv'));
writetable(buildFeatureSummary(T, numVars), fullfile(outDir, 'eda_feature_summary.csv'));
Gs = buildGenreSummary(T);
writetable(Gs, fullfile(outDir, 'eda_genre_summary.csv'));
disp(Gs);

%% ---- Fig. 1 for the paper: four panels ----
fig = newFig(W2, 4.6);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); plotClassBalance(ax, T, col);  title(ax, '(a) Class balance', 'FontWeight', 'normal');
ax = nexttile(tl); plotGenreRate(ax, T, col);     title(ax, '(b) Viral rate by genre', 'FontWeight', 'normal');
ax = nexttile(tl); plotFeatureCorr(ax, T, numVars, col); title(ax, '(c) Correlation with viral', 'FontWeight', 'normal');
ax = nexttile(tl); plotAcousticDist(ax, T, col);  title(ax, '(d) Acousticness by class', 'FontWeight', 'normal');
saveFig(fig, outDir, 'fig1_eda_overview');

%% ---- Single-purpose figures (slides, backup, or paper alternates) ----
fig = newFig(W1, 2.3); ax = axes(fig); plotClassBalance(ax, T, col);
saveFig(fig, outDir, 'eda_class_balance');

fig = newFig(W1, 2.8); ax = axes(fig); plotGenreRate(ax, T, col);
saveFig(fig, outDir, 'eda_genre_rate');

% Feature distributions by class (10 features; release_year is near-uniform, see year plot)
feat  = numVars(2:end);
label = {'duration (s)','danceability','energy','valence','tempo (BPM)', ...
         'acousticness','instrumentalness','speechiness','liveness','loudness (dB)'};
fig = newFig(W2, 3.4);
tl = tiledlayout(fig, 2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(feat)
    ax = nexttile(tl); hold(ax, 'on');
    x = T.(feat{k});
    e = linspace(min(x), max(x), 26);
    histogram(ax, x(T.viral==0), e, 'Normalization', 'probability', 'FaceColor', col.non,   'EdgeColor', 'none', 'FaceAlpha', 0.95);
    histogram(ax, x(T.viral==1), e, 'Normalization', 'probability', 'FaceColor', col.viral, 'EdgeColor', 'none', 'FaceAlpha', 0.60);
    title(ax, label{k}, 'FontWeight', 'normal');
    if mod(k, 5) == 1, ylabel(ax, 'Proportion'); end
    if k == 1, legend(ax, {'Non-viral','Viral'}, 'Box', 'off', 'Location', 'northeast'); end
    styleAx(ax);
end
saveFig(fig, outDir, 'eda_feature_dist');

% Correlation heatmap (numbers printed in each cell so it reads in grayscale)
fig = newFig(W2, 4.4); ax = axes(fig);
plotCorrHeatmap(ax, T, numVars);
saveFig(fig, outDir, 'eda_correlation');

% Skew before / after log1p
sv = {'acousticness','instrumentalness','speechiness'};
fig = newFig(W2, 3.2);
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:2
    for k = 1:3
        ax = nexttile(tl);
        x = T.(sv{k});
        if r == 2, x = log1p(x); ttl = ['log1p(' sv{k} ')']; else, ttl = sv{k}; end
        histogram(ax, x, 30, 'FaceColor', col.viral, 'EdgeColor', 'none');
        title(ax, ttl, 'FontWeight', 'normal');
        xlabel(ax, sprintf('skewness = %.2f', skewness(x)));
        if k == 1, ylabel(ax, 'Tracks'); end
        styleAx(ax);
    end
end
saveFig(fig, outDir, 'eda_skew_log1p');

% Viral rate by release year
fig = newFig(W1, 2.3); ax = axes(fig); hold(ax, 'on');
[Gy, yrs] = findgroups(T.release_year);
ry = splitapply(@mean, T.viral, Gy);
plot(ax, yrs, 100*ry, '-o', 'Color', col.viral, 'MarkerFaceColor', col.viral, 'MarkerSize', 3, 'LineWidth', 1);
yline(ax, 100*mean(T.viral), '--k', 'LineWidth', 0.75);
xlabel(ax, 'Release year'); ylabel(ax, 'Viral rate (%)'); ylim(ax, [0 60]); xlim(ax, [2010 2025]);
styleAx(ax);
saveFig(fig, outDir, 'eda_year_rate');

% Why streams is excluded: it separates the classes because it is an outcome
fig = newFig(W1, 2.5); ax = axes(fig);
boxchart(ax, categorical(T.viral, [0 1], {'Non-viral','Viral'}), log10(T.streams), ...
    'BoxFaceColor', col.viral, 'MarkerColor', [0.4 0.4 0.4], 'MarkerSize', 2);
ylabel(ax, 'log_{10}(streams)');
styleAx(ax);
saveFig(fig, outDir, 'eda_streams_leak');

fprintf('\nFigures and tables written to: %s\n', fullfile(pwd, outDir));
end


%% ======================================================================
%  Plot helpers (each draws into a supplied axes so panels can be reused)
%  ======================================================================

function plotClassBalance(ax, T, col)
n1 = sum(T.viral == 1); n0 = sum(T.viral == 0); N = n0 + n1;
b = bar(ax, [1 2], [n0 n1], 0.6, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.5);
b.CData = [col.non; col.viral];
set(ax, 'XTick', [1 2], 'XTickLabel', {'Non-viral (0)','Viral (1)'});
ylabel(ax, 'Number of tracks');
ylim(ax, [0 1.2*max(n0, n1)]);
text(ax, [1 2], [n0 n1], {sprintf('%d (%.1f%%)', n0, 100*n0/N), sprintf('%d (%.1f%%)', n1, 100*n1/N)}, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'FontName', 'Times New Roman');
styleAx(ax);
end

function plotGenreRate(ax, T, col)
[G, genres] = findgroups(string(T.genre));
rate = 100 * splitapply(@mean, T.viral, G);
[rate, ix] = sort(rate, 'ascend');
genres = genres(ix);
barh(ax, rate, 0.7, 'FaceColor', col.viral, 'EdgeColor', 'k', 'LineWidth', 0.5);
set(ax, 'YTick', 1:numel(genres), 'YTickLabel', cellstr(genres));
hold(ax, 'on');
xline(ax, 100*mean(T.viral), '--k', 'LineWidth', 0.75);      % overall viral rate
text(ax, rate + 1.5, 1:numel(rate), cellstr(compose('%.1f', rate)), ...
    'FontSize', 8, 'FontName', 'Times New Roman', 'VerticalAlignment', 'middle');
xlabel(ax, 'Viral rate (%)'); xlim(ax, [0 80]);
styleAx(ax);
end

function plotFeatureCorr(ax, T, numVars, col)
r = corr(table2array(T(:, numVars)), T.viral);        % point-biserial correlation
short = shortNames(numVars);
[~, ix] = sort(r, 'ascend');
barh(ax, r(ix), 0.7, 'FaceColor', col.viral, 'EdgeColor', 'k', 'LineWidth', 0.5);
set(ax, 'YTick', 1:numel(ix), 'YTickLabel', short(ix));
xline(ax, 0, 'k', 'LineWidth', 0.5);
xlabel(ax, 'Pearson r with viral');
xlim(ax, [-0.5 0.5]);
styleAx(ax);
end

function plotAcousticDist(ax, T, col)
hold(ax, 'on');
e = linspace(0, 1, 21);
histogram(ax, T.acousticness(T.viral==0), e, 'Normalization', 'probability', 'FaceColor', col.non,   'EdgeColor', 'k', 'LineWidth', 0.25);
histogram(ax, T.acousticness(T.viral==1), e, 'Normalization', 'probability', 'FaceColor', col.viral, 'EdgeColor', 'k', 'LineWidth', 0.25, 'FaceAlpha', 0.65);
xlabel(ax, 'Acousticness (0-1)'); ylabel(ax, 'Proportion of class');
legend(ax, {'Non-viral','Viral'}, 'Box', 'off', 'Location', 'northeast');
styleAx(ax);
end

function plotCorrHeatmap(ax, T, numVars)
M = [table2array(T(:, numVars)), T.viral];
R = corrcoef(M);
n = size(R, 1);
imagesc(ax, R, [-1 1]);
axis(ax, 'image');
colormap(ax, blueWhiteRed());
cb = colorbar(ax); cb.Label.String = 'Pearson r'; cb.FontName = 'Times New Roman'; cb.FontSize = 8;
labels = [shortNames(numVars); {'viral'}];
set(ax, 'XTick', 1:n, 'XTickLabel', labels, 'XTickLabelRotation', 45, ...
        'YTick', 1:n, 'YTickLabel', labels, 'TickLength', [0 0]);
for i = 1:n
    for j = 1:n
        if abs(R(i, j)) > 0.5, tc = 'w'; else, tc = 'k'; end
        text(ax, j, i, sprintf('%.2f', R(i, j)), 'HorizontalAlignment', 'center', ...
            'FontSize', 8, 'FontName', 'Times New Roman', 'Color', tc);
    end
end
styleAx(ax);
end

function s = shortNames(numVars)
map = containers.Map( ...
    {'release_year','duration_sec','danceability','energy','valence','tempo_bpm', ...
     'acousticness','instrumentalness','speechiness','liveness','loudness_db'}, ...
    {'year','duration','danceability','energy','valence','tempo', ...
     'acousticness','instrumental.','speechiness','liveness','loudness'});
s = cellfun(@(v) map(v), numVars(:), 'UniformOutput', false);
end

function cmap = blueWhiteRed()
up = linspace(0.17, 1, 128)';
dn = linspace(1, 0.15, 128)';
cmap = [up, up + (1 - up) * 0.2, ones(128, 1); ones(128, 1), dn + (1 - dn) * 0.0, dn];
cmap = min(max(cmap, 0), 1);
end

function fig = newFig(w, h)
fig = figure('Units', 'inches', 'Position', [1 1 w h], 'Color', 'w', 'Visible', 'off');
set(fig, 'PaperUnits', 'inches', 'PaperSize', [w h], 'PaperPosition', [0 0 w h]);
end

function styleAx(ax)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8, 'LineWidth', 0.5, ...
        'TickDir', 'out', 'Box', 'off');
end

function saveFig(fig, outDir, name)
set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');
exportgraphics(fig, fullfile(outDir, string(name) + ".pdf"), 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(outDir, string(name) + ".png"), 'Resolution', 300, 'BackgroundColor', 'white');
close(fig);
end


%% ======================================================================
%  Tables
%  ======================================================================

function D = buildDictionary(T)
% Paper TABLE II: data dictionary. Ranges are computed from the data.
rg = @(v, u) sprintf('%s to %s%s', num2str(min(v)), num2str(max(v)), u);
genres = categories(T.genre);

Variable = ["track_id";"track_name";"artist";"genre";"release_year";"duration_sec"; ...
            "danceability";"energy";"valence";"tempo_bpm";"acousticness"; ...
            "instrumentalness";"speechiness";"liveness";"loudness_db";"streams";"viral"];
Type = ["string";"string";"string";"categorical";"integer";"integer"; ...
        "float";"float";"float";"float";"float";"float";"float";"float";"float";"integer";"binary"];
Range = [ ...
    "track_00001 to track_" + sprintf('%05d', height(T)); ...
    "Track_00001 to Track_" + sprintf('%05d', height(T)); ...
    "Artist_xxxx (synthetic)"; ...
    string(numel(genres)) + " genres"; ...
    string(rg(T.release_year, "")); ...
    string(rg(T.duration_sec, " s")); ...
    string(rg(T.danceability, "")); string(rg(T.energy, "")); string(rg(T.valence, "")); ...
    string(rg(T.tempo_bpm, " BPM")); string(rg(T.acousticness, "")); ...
    string(rg(T.instrumentalness, "")); string(rg(T.speechiness, "")); ...
    string(rg(T.liveness, "")); string(rg(T.loudness_db, " dB")); ...
    string(rg(T.streams, "")); "0 or 1"];
Description = [ ...
    "Unique track identifier"; "Synthetic track name"; "Synthetic artist name"; ...
    "Primary genre"; "Year released"; "Track length"; ...
    "Suitability for dancing"; "Intensity and activity"; "Musical positivity"; ...
    "Beats per minute"; "Confidence the track is acoustic"; "Likelihood of no vocals"; ...
    "Presence of spoken words"; "Presence of an audience"; "Overall loudness"; ...
    "Total play count"; "1 if viral, 0 otherwise"];
Role = ["Dropped (identifier)";"Dropped (identifier)";"Dropped (identifier)"; ...
        "Predictor (one-hot)";"Predictor (tested with and without)";"Predictor"; ...
        "Predictor";"Predictor";"Predictor";"Predictor"; ...
        "Predictor (log1p)";"Predictor (log1p)";"Predictor (log1p)";"Predictor";"Predictor"; ...
        "Excluded (post-outcome, leaks target)";"Target"];
D = table(Variable, Type, Range, Description, Role);
end

function S = buildFeatureSummary(T, numVars)
nV = numel(numVars);
Feature = string(numVars(:));
Mean = zeros(nV, 1); SD = Mean; Min = Mean; Max = Mean; Skew = Mean;
ZeroPct = Mean; IQROutlierPct = Mean; CorrWithViral = Mean;
for j = 1:nV
    x = T.(numVars{j});
    Mean(j) = mean(x); SD(j) = std(x); Min(j) = min(x); Max(j) = max(x);
    Skew(j) = skewness(x);
    ZeroPct(j) = 100 * mean(x == 0);
    IQROutlierPct(j) = 100 * mean(isoutlier(x, 'quartiles'));
    CorrWithViral(j) = corr(x, T.viral);
end
S = table(Feature, Mean, SD, Min, Max, Skew, ZeroPct, IQROutlierPct, CorrWithViral);
end

function S = buildGenreSummary(T)
[G, Genre] = findgroups(string(T.genre));
Tracks = splitapply(@numel, T.viral, G);
ViralTracks = splitapply(@sum, T.viral, G);
ViralRatePct = 100 * ViralTracks ./ Tracks;
S = sortrows(table(Genre, Tracks, ViralTracks, ViralRatePct), 'ViralRatePct', 'descend');
end