# CNN Multihead vs Binary Head Comparison at Binary FPR 0.1

This comparison keeps only the winner CNN from each run and reports inference-set results. The binary-head class metrics use a threshold selected for accepted FPR `0.1` with the pipeline rule `maximize_recall_subject_to_fpr`.

Source runs:

- Binary head: `runs/supervised/cnn_ag_news_imdb_attack_family_binary_tuning`
- Multihead/multiclass head: `runs/supervised/cnn_ag_news_imdb_attack_family_multiclass_tuning_4`

## Classification Strategy

The binary-head CNN is trained with two labels, `clean` and `backdoored`. At inference it emits a single backdoor score using the `positive_class_score` projection; AUROC metrics rank samples by this score. For the thresholded metrics in this report, a sample is classified as backdoored when `score >= 0.838159327355234`. Because this model does not predict attack families, the RIPPLE, insertsent, stybkd, and syntactic rows are attack-vs-shared-clean detection slices rather than true attack-family classifications.

The multihead/multiclass CNN is trained with five output classes: `clean`, `RIPPLE`, `insertsent`, `stybkd`, and `syntactic`. At inference, the predicted class is the argmax over those class probabilities. For binary detection, the same model is projected to a backdoor score with `1 - P(clean)`, so it can be evaluated both as a clean-vs-backdoor detector and as a direct attack-family classifier.

## Winner Models

| Strategy | Winner task | Selection metric | CV score | Winner hyperparameters |
| --- | ---: | --- | ---: | --- |
| Binary head | 32 | AUROC | 0.983 +/- 0.003 | `conv_channels=384`, `num_conv_layers=3`, `kernel_size=5`, `dropout=0.0`, `learning_rate=0.0001`, `weight_decay=0.0` |
| Multihead/multiclass | 20 | Macro F1 | 0.915 +/- 0.013 | `conv_channels=256`, `num_conv_layers=4`, `kernel_size=5`, `dropout=0.0`, `learning_rate=0.0001`, `weight_decay=0.0` |

## Binary FPR 0.1 Threshold

The binary CNN run does not contain a saved `selected_threshold.json` because it was run without a calibration split. The threshold below was therefore recomputed from `train_scores.csv` using the same pipeline selection rule.

| Source | Accepted FPR | Selected threshold | Source precision | Source recall | Source FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary-head train scores | 0.100 | 0.838159 | 1.000 | 1.000 | 0.000 |

Applied to the inference set, this threshold gives binary accuracy `0.938`, precision `0.962`, recall `0.960`, and FPR `0.150`.

## Binary Detection

| Strategy | Binary projection | Inference AUROC | Inference AUPRC | P@P |
| --- | --- | ---: | ---: | ---: |
| Binary head | `positive_class_score` | 0.977 | 0.994 | 0.963 |
| Multihead/multiclass | `one_minus_clean_probability` | 0.986 | 0.996 | 0.968 |

The multihead winner has the stronger clean-vs-backdoor AUROC on the inference set. The threshold choice affects precision, recall, and accuracy, but not AUROC.

## Per-Attack AUROC

| Attack | Binary head AUROC | Multihead AUROC | Better |
| --- | ---: | ---: | --- |
| RIPPLE | 0.988 | 0.996 | Multihead |
| insertsent | 0.967 | 0.981 | Multihead |
| stybkd | 0.988 | 0.994 | Multihead |
| syntactic | 0.967 | 0.972 | Multihead |

## Class Metrics

For the multihead model, accuracy is one-vs-rest accuracy on the full 500-sample inference set, and precision/recall are direct multiclass argmax metrics. Overall multiclass accuracy is 0.900.

For the binary head, `clean` is measured on the full inference set. Attack rows are measured on each attack plus the shared clean pool using `score >= 0.838159327355234`; these rows answer "was this attack detected as backdoored?" rather than "was this attack family named correctly?"

| Class | Binary acc. | Binary precision | Binary recall | Multihead acc. | Multihead precision | Multihead recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.938 | 0.842 | 0.850 | 0.948 | 0.870 | 0.870 |
| RIPPLE | 0.925 | 0.870 | 1.000 | 0.986 | 0.951 | 0.980 |
| insertsent | 0.895 | 0.862 | 0.940 | 0.948 | 0.894 | 0.840 |
| stybkd | 0.910 | 0.866 | 0.970 | 0.966 | 0.881 | 0.960 |
| syntactic | 0.890 | 0.861 | 0.930 | 0.952 | 0.904 | 0.850 |

## Takeaway

With the binary-head threshold selected for accepted FPR `0.1`, the binary model trades some attack recall for better clean recall compared with the default `0.5` threshold report. The multihead CNN still wins on threshold-free binary AUROC and every per-attack AUROC, and it remains the only strategy that produces actual attack-family labels.
