#!/bin/bash
#SBATCH --job-name=cluster_z_space
#SBATCH --output=logs/cluster_z_space_%j.log
#SBATCH --error=logs/cluster_z_space_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load conda
eval "$(conda shell.bash hook)"
conda activate upeftg

Run unsupervised baselines
python3 cluster_z_space.py \
  --data-dir processed_data_120+120 \
  --n-components 30 \
  --output-dir clustering_results_120+120 \
  --algorithms kmeans hierarchical dbscan gmm mahalanobis isolation_forest lof \
  --k-list 2 3 4 5 \
  --eps-list 0.5 1.0 1.5 2.0 \
  --min-samples 2 \
  --selection-metric silhouette \
  --use-offline-label-metrics

# python cluster_z_space.py \
#   --data-dir delta_features \
#   --feature-file delta_features/delta_singular_values.npy \
#   --output-dir clustering_results_delta_sv \
#   --algorithms gmm mahalanobis isolation_forest lof \
#   --selection-metric stability \
#   --use-offline-label-metrics

# python cluster_z_space.py \
#   --data-dir delta_features \
#   --feature-file delta_features/delta_frobenius.npy \
#   --output-dir clustering_results_delta_fro \
#   --algorithms gmm mahalanobis isolation_forest lof \
#   --selection-metric stability \
#   --use-offline-label-metrics


# List output files
echo "Clustering complete! Output files:"
ls -lh clustering_results/
