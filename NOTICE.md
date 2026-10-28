# Copyright and source notices

## Project additions

Copyright (c) 2026 the respective Engineering Big Data Group 1 contributors.

The project additions in `cnn_baseline/`, `scripts/`, `tools/`, and the accompanying project documentation are distributed under the GNU General Public License, version 3 (GPL-3.0-only), except where an existing third-party notice specifies otherwise. See the root `LICENSE`.

## RoNIN

This repository includes the provided RoNIN source snapshot in `ronin-master/`. Its original notices and GPL v3 license are retained. Project training and evaluation adapters reuse its ResNet1D model and trajectory metric implementation. The upstream repository is https://github.com/Sachini/ronin .

Reference: Sachini Herath, Hang Yan, and Yasutaka Furukawa. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods. ICRA 2020. See the upstream README for the paper and dataset links.

The original files are preserved. The added NPZ loading, window-mean velocity training/evaluation, CNN/GRU/attention models, rotation experiments, and packaging utilities are identified by their separate project directories. This distribution snapshot was prepared on 2026-10-28.

## Data and model artifacts

Processed datasets, reference trajectories, trained checkpoints, and complete per-window predictions are excluded from this repository. Their absence does not grant redistribution rights in any separately obtained files. Aggregate evaluation tables and summary figures report the experiments described in the project README.
