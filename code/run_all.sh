#!/bin/bash
set -e

echo "=== Running all experiments ==="

python analysis/batch_scan.py
python analysis/local_pca_analysis.py
python analysis/cross_model_analysis.py
python analysis/head11_deep_analysis.py
python analysis/layer11_analysis.py
python analysis/ablation_v2.py
python analysis/cumulative_ablation.py
python analysis/domain_width_experiment.py

echo "=== Generating appendix figures ==="

python utils/plot_figA1.py
python utils/plot_figA2.py
python utils/plot_figA3.py
python utils/plot_figA4.py
python utils/crop_fig01_fig02.py

echo "=== All done ==="