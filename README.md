# rosneuro_filters_car

ROS-Neuro plugin implementing a **Common Average Reference (CAR)** spatial filter. For each sample it computes the mean across non-excluded EEG channels and subtracts it from all channels.

---

## 1. Algorithm

For each sample in an input chunk `[nsamples × nchannels]`:

1. Identify the set of *valid* channels (all channels minus EOG exclusions).
2. Compute their mean → scalar baseline.
3. Subtract the baseline from **all** channels (including EOG channels).

Excluding noisy channels (e.g. EOG) from the average prevents blink artefacts from contaminating the reference baseline.

---

## 2. Configuration

Channel exclusion can be specified by **name** (recommended) or by **1-based index** (legacy).

### Name-based (recommended)

```yaml
CarCfg:
  name: car
  type: CarFilterDouble
  params:
    EOG_ch_names: ['Fp1', 'Fp2']   # matched case-insensitively against eeg.info.labels
```

Names are resolved to indices at runtime when channel labels become available (see `configure(ch_labels, eog_names)` below).

### Index-based (legacy)

```yaml
CarCfg:
  name: car
  type: CarFilterDouble
  params:
    EOG_ch: [1, 2]   # 1-based indexing
```

If neither parameter is present, CAR is applied to all channels (with a ROS_WARN).

---

## 3. API

### Plugin (`pluginlib`)

```cpp
rosneuro::Car<double> car;

// Load from ROS param server (EOG_ch_names or EOG_ch)
car.configure("CarCfg");

// Resolve names → indices once channel labels are known (e.g. from first NeuroFrame)
car.configure(ch_labels, eog_names);   // vector<string>, vector<string>

// Legacy: set indices directly (0-based)
car.configure(std::vector<int>{0, 1});

// Apply to a chunk [nsamples × nchannels]
DynamicMatrix<double> out = car.apply(in);
```

### `configure()` variants

| Signature | Description |
|-----------|-------------|
| `configure()` | Reads `EOG_ch_names` (strings) or `EOG_ch` (1-based ints) from the ROS param server |
| `configure(const string& param_name)` | Delegates to `Filter<T>::configure(param_name)` to set the param namespace |
| `configure(ch_labels, eog_names)` | Resolves string names to 0-based indices via case-insensitive match |
| `configure(const vector<int>& eog_ch)` | Sets 0-based exclusion indices directly |

---

## 4. Testing

### 4a. CSV-based test (quick sanity check)

Replays `test/rawdata.csv` through the full ROS pipeline.

```bash
roslaunch rosneuro_filters_car test_node_car.launch
# Wait until publisher finishes, then Ctrl+C.
```

Produces `test/car_processing.csv` and `test/car_processing_first_seq.txt`.

Compare with MATLAB:

```matlab
input_mode = 'csv';
test_car   % from workspace root
```

### 4b. GDF-based test (end-to-end with real acquisition)

Replays `test/prova32ch.gdf` (512 Hz, 32 channels) through `rosneuro_acquisition` with the eegdev `datafile` plugin.

```bash
roslaunch rosneuro_filters_car test_node_car_gdf.launch \
    gdf_file:=$(rospack find rosneuro_filters_car)/test/prova32ch.gdf \
    samplerate:=512 \
    framerate:=16
# Wait for the file to finish, then Ctrl+C.
```

Produces in `test/`:
- `car_gdf_output.csv` — CAR-filtered EEG indexed by arrival order
- `car_gdf_output_first_seq.txt` — first seq received (startup frame loss)

Compare with MATLAB:

```matlab
input_mode = 'gdf';
test_car   % from workspace root
```

### Alignment details

**`first_seq` — startup frame loss**: `rosneuro_acquisition` may miss the first 1–2 frames. The logger records the first seq received. MATLAB starts its loop at `seq = first_seq` so both pipelines start from the same sample.

**Acquisition pipeline delay (GDF only)**: the eegdev `datafile` plugin buffers one frame internally before publishing. The MATLAB script detects this with `xcorr` on one channel and corrects automatically. Two figures are produced:
- **RAW** — unaligned comparison (shows the lag)
- **ALIGNED** — lag-corrected comparison (traces should overlap)

### ROS pipeline

```
test/rawdata.csv (CSV) or test/prova32ch.gdf (GDF)
  → /neurodata  (rosneuro_msgs/NeuroFrame)
      → car_node  (resolves EOG_ch_names on first frame, applies CAR)
          → /car_output  (rosneuro_msgs/NeuroFrame, filtered)
              → car_logger  → test/car_*.csv + test/car_*_first_seq.txt
```

---

## 5. Dependencies

| Library | Used for |
|---------|---------|
| `rosneuro_filters` | `Filter<T>` base class, pluginlib integration |
| `rosneuro_msgs` | `NeuroFrame` message (test nodes only) |
| `Eigen3` | Matrix operations |
| `pluginlib` | Dynamic plugin loading |
