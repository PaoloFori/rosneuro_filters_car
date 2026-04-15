# ROS-Neuro CAR filter

This ROS-Neuro filter plugin implements a Common Average Reference filter to remove the average component from all data channels.

## Algorithm:
The filter applies the Common Average Reference (CAR) spatial filter to the data. 
For each sample, it computes a single "common average" baseline across the EEG channels. 
Crucially, if auxiliary channels represent noise (e.g., EOG channels), the user can provide an `EOG_ch` parameter. The node will rigorously **exclude those specific channels from the average computation** so that large noise spikes (like blinks) do not pollute the reference baseline. Once the pure baseline is computed, it is subtracted from all channels.

## YAML configuration
The configuration file accepts the standard node parameters along with the optional `EOG_ch` list (using standard 1-based indexing).

```yaml
CarCfgTest:
  name: car
  type: CarFilterFloat
  params:
    EOG_ch: [33, 34, 35] # FP1, FP2, EOG
```
*If `EOG_ch` is not provided, the filter computes the average using all available channels across the input matrix.*
