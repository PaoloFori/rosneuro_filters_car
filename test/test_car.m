%% CAR MATLAB simulation
% Replicates the ROS car_node chunk by chunk and compares with its output.
%
% KEY alignment principle:
%   ROS starts processing at seq = first_seq (first frame actually received).
%   MATLAB must also start at first_seq.  For the GDF input an additional
%   1-frame acquisition pipeline delay is measured via cross-correlation and
%   corrected automatically.
%
% Workflow (CSV):
%   1. roslaunch rosneuro_filters_car test_node_car.launch
%   2. Ctrl+C → car_processing.csv + car_processing_first_seq.txt
%   3. Run this script with input_mode = 'csv'.
%
% Workflow (GDF):
%   1. roslaunch rosneuro_filters_car test_node_car_gdf.launch \
%        gdf_file:=$(rospack find rosneuro_filters_car)/test/prova32ch.gdf
%   2. Ctrl+C → car_gdf_output.csv + car_gdf_output_first_seq.txt
%   3. Run this script with input_mode = 'gdf'.

clear all; clc; close all;

%% --- input mode ---
input_mode = 'csv';   % 'gdf' | 'csv'
ch_plot = 5;

%% --- paths ---
data_dir = './test_node_data/';
out_dir  = './test_node_data/rosneuro_filters_car/';
car_yaml = './src/rosneuro_filters_car/cfg/car.yaml';

if strcmp(input_mode, 'gdf')
    input_file   = [data_dir 'prova32ch.gdf'];
    ros_file     = [out_dir  'car_gdf_output.csv'];
    framerate    = 16;
    plot_start_s = 2;    % skip first N seconds in plots ([] = show all)
else
    input_file   = [data_dir 'raw_eeg_32ch.csv'];
    ros_file     = [out_dir  'car_processing.csv'];
    framerate    = 20;
    plot_start_s = [];
end

%% --- read first_seq ---
first_seq_file = strrep(ros_file, '.csv', '_first_seq.txt');
first_seq = 0;
if isfile(first_seq_file)
    first_seq = readmatrix(first_seq_file);
    fprintf('ROS first_seq = %d (lost %d frame(s) at startup)\n', first_seq, first_seq);
else
    fprintf('first_seq file not found – assuming first_seq = 0.\n');
end

%% --- load CAR config ---
car_cfg      = yaml.ReadYaml(car_yaml);
eog_ch_names = car_cfg.CarCfg.params.EOG_ch_names;
default_eog_ch_names = [{'Fp1'}, {'Fp2'}];
if ~iscell(eog_ch_names)
    warning('EOG_ch_names worng. Defaul values are used: ');
    disp(default_eog_ch_names)
    eog_ch_names = default_eog_ch_names;
end

%% --- load data ---
[~, ~, ext] = fileparts(input_file);
if strcmpi(ext, '.gdf')
    [data_raw, hdr] = sload(input_file);   % BIOSIG required
    sampleRate = hdr.SampleRate;
    ch_names   = cellstr(hdr.Label);
    n_eeg = sum(~cellfun(@(c) contains(lower(c), {'status','trigger','mkr'}), ch_names));
    data     = data_raw(:, 1:n_eeg);
    ch_names = ch_names(1:n_eeg);
    fprintf('GDF: %d samples x %d EEG channels @ %.0f Hz\n', size(data,1), n_eeg, sampleRate);
else
    data       = readmatrix(input_file);
    sampleRate = 500;
    ch_names   = {'Fp1','Fp2','F3','Fz','F4','FC1','FC2','C3','Cz','C4', ...
                  'CP1','CP2','P3','Pz','P4','POz','O1','O2','CPz','F1', ...
                  'F2','FC5','FCz','FC6','C1','C2','CP5','CP6','P5','P1','P2','P6'};
    ch_names   = ch_names(1:size(data,2));
    fprintf('CSV: %d samples x %d channels @ %.0f Hz\n', size(data,1), size(data,2), sampleRate);
end

% Cast to single to match the float32 data that the publisher sends over ROS.
% Without this, MATLAB computes CAR on float64 while ROS uses float32,
% causing systematic differences for large-amplitude raw ADC values.
data = single(data);

nchannels = size(data, 2);
chunkSize = round(sampleRate / framerate);
n_frames  = floor(size(data, 1) / chunkSize);

fprintf('  chunkSize : %d samples\n', chunkSize);

%% --- resolve EOG channels (1-based) ---
EOG_ch = zeros(1, numel(eog_ch_names));
for k = 1:numel(eog_ch_names)
    m = find(strcmpi(ch_names, eog_ch_names{k}), 1);
    if isempty(m)
        error('EOG channel "%s" not found.\nAvailable: %s', eog_ch_names{k}, strjoin(ch_names, ', '));
    end
    EOG_ch(k) = m;
end
non_eog_ch = setdiff(1:nchannels, EOG_ch);
fprintf('EOG channels: [%s] → indices [%s]\n', strjoin(eog_ch_names, ', '), num2str(EOG_ch));

%% --- apply CAR chunk by chunk (starting at first_seq) ---
n_output_frames  = n_frames - first_seq;
matlab_output    = zeros(n_output_frames * chunkSize, nchannels);

for seq = first_seq : n_frames - 1
    idx_in  = seq * chunkSize + 1 : (seq + 1) * chunkSize;
    f_out   = seq - first_seq + 1;   % 1-based output frame
    idx_out = (f_out - 1) * chunkSize + 1 : f_out * chunkSize;

    chunk    = data(idx_in, :);
    car_mean = mean(chunk(:, non_eog_ch), 2);
    matlab_output(idx_out, :) = chunk - car_mean;
end

%% --- compare with ROS output ---
if ~isfile(ros_file)
    warning('ROS output not found: %s', ros_file);
    t = (0 : size(matlab_output,1)-1) / sampleRate;
    figure; plot(t, matlab_output(:, ch_plot)); xlabel('time [s]');
    ylabel('amplitude'); title('MATLAB only (no ROS ref)'); grid on;
    return;
end

ros_data = readmatrix(ros_file);   % [n_output_samples x nchannels]

n_compare = min(size(ros_data, 1), size(matlab_output, 1));
ros_out   = ros_data(1:n_compare, :);
mat_out   = matlab_output(1:n_compare, :);

%% --- cross-correlation on one channel to measure lag ---
% GDF mode: eegdev buffers one frame internally → expected lag is 1 frame.
% CSV mode: publisher waits for subscriber, so lag is 0 or 1 frame at most.
%   Search window is kept to ±1 frame to avoid spurious xcorr peaks at
%   alpha/beta harmonic lags (e.g. 10 Hz × 50ms/frame = 2 frames apart).
ch_xcorr = 3;
MAX_LAG_SEARCH = chunkSize;   % ±1 frame for both modes; enough for CSV
if strcmp(input_mode, 'gdf')
    MAX_LAG_SEARCH = 5 * chunkSize;
end

[xcf, lags] = xcorr(ros_out(:,ch_xcorr) - mean(ros_out(:,ch_xcorr)), ...
                    mat_out(:,ch_xcorr) - mean(mat_out(:,ch_xcorr)), ...
                    MAX_LAG_SEARCH, 'normalized');
[~, peak_idx]     = max(xcf);
measured_lag_samp = lags(peak_idx);
measured_lag_frames = round(measured_lag_samp / chunkSize);
fprintf('Lag: %+d samples (%+d frame(s))  ', measured_lag_samp, measured_lag_frames);
if measured_lag_samp == 0
    fprintf('[no residual lag]\n');
elseif measured_lag_samp > 0
    fprintf('[ROS lags MATLAB]\n');
else
    fprintf('[MATLAB lags ROS]\n');
end

%% --- apply lag correction ---
if measured_lag_samp > 0
    r_aligned = ros_out(1 + measured_lag_samp : end, :);
    m_aligned = mat_out(1 : end - measured_lag_samp, :);
elseif measured_lag_samp < 0
    shift     = -measured_lag_samp;
    r_aligned = ros_out(1 : end - shift, :);
    m_aligned = mat_out(1 + shift : end, :);
else
    r_aligned = ros_out;
    m_aligned = mat_out;
end

%% --- restrict window for plots/stats ---
if ~isempty(plot_start_s)
    skip_samp    = round(plot_start_s * sampleRate);
    raw_skip     = min(skip_samp, size(ros_out, 1));
    aligned_skip = min(skip_samp, size(r_aligned, 1));
else
    raw_skip     = 0;
    aligned_skip = 0;
end

t_raw     = (0 : n_compare - 1) / sampleRate;
t_aligned = (0 : size(r_aligned, 1) - 1) / sampleRate;

r_raw_plot = ros_out(raw_skip+1 : end, :);
m_raw_plot = mat_out(raw_skip+1 : end, :);
t_raw_plot = t_raw(raw_skip+1 : end);

r_al_plot  = r_aligned(aligned_skip+1 : end, :);
m_al_plot  = m_aligned(aligned_skip+1 : end, :);
t_al_plot  = t_aligned(aligned_skip+1 : end);

%% --- metrics ---
mae_raw     = mean(abs(r_raw_plot(:, ch_plot) - m_raw_plot(:, ch_plot)));
mae_aligned = mean(abs(r_al_plot(:, ch_plot)  - m_al_plot(:, ch_plot)));

if ~isempty(plot_start_s)
    fprintf('Plotting from %.1f s onward (%d raw / %d aligned samples shown)\n', ...
            plot_start_s, size(r_raw_plot,1), size(r_al_plot,1));
end
fprintf('MAE ch%d (raw)     : %.6f\n', ch_plot, mae_raw);
fprintf('MAE ch%d (aligned) : %.6f\n', ch_plot, mae_aligned);

%% --- plot: raw ---
figure;
subplot(2,1,1); hold on;
plot(t_raw_plot, r_raw_plot(:, ch_plot), 'b',   'LineWidth', 1.5);
plot(t_raw_plot, m_raw_plot(:, ch_plot), 'r--', 'LineWidth', 1);
legend('ROS node', 'MATLAB simulation');
ylabel('amplitude'); grid on; hold off;
title(sprintf('[RAW] CAR | ch=%s | first seq=%d', ch_names{ch_plot}, first_seq));
subplot(2,1,2);
bar(t_raw_plot, abs(r_raw_plot(:,ch_plot) - m_raw_plot(:,ch_plot)));
xlabel('time [s]'); ylabel('|diff|');
title(sprintf('Differences (lag=%+d samp)', measured_lag_samp)); grid on;

%% --- plot: lag-corrected ---
figure;
subplot(2,1,1); hold on;
plot(t_al_plot, r_al_plot(:, ch_plot), 'b',   'LineWidth', 1.5);
plot(t_al_plot, m_al_plot(:, ch_plot), 'r--', 'LineWidth', 1);
legend('ROS node', 'MATLAB simulation');
ylabel('amplitude'); grid on; hold off;
title(sprintf('[ALIGNED lag=%+d samp] CAR | ch=%s', measured_lag_samp, ch_names{ch_plot}));
subplot(2,1,2);
bar(t_al_plot, abs(r_al_plot(:,ch_plot) - m_al_plot(:,ch_plot)));
xlabel('time [s]'); ylabel('|diff|'); title('Differences after alignment'); grid on;
