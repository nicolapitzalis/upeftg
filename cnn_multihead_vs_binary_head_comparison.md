# CNN Multihead vs Binary Head Comparison

This comparison keeps only the winner CNN from each run and reports inference-set results. Source runs:

- Binary head: `runs/supervised/cnn_ag_news_imdb_attack_family_binary_tuning`
- Multihead/multiclass head: `runs/supervised/cnn_ag_news_imdb_attack_family_multiclass_tuning_4`

## Classification Strategy

The binary-head CNN is trained with two labels, `clean` and `backdoored`. At inference it emits a single backdoor score using the `positive_class_score` projection; binary AUROC and per-attack AUROC rank samples by this score. Thresholded class metrics below use the natural probability decision rule `score >= 0.5` for `backdoored`. Because this model does not predict attack families, the RIPPLE, insertsent, stybkd, and syntactic rows are attack-vs-shared-clean detection slices rather than true attack-family classifications.

The multihead/multiclass CNN is trained with five output classes: `clean`, `RIPPLE`, `insertsent`, `stybkd`, and `syntactic`. At inference, the predicted class is the argmax over those class probabilities. For binary detection, the same model is projected to a backdoor score with `1 - P(clean)`, so it can be evaluated both as a clean-vs-backdoor detector and as a direct attack-family classifier.

## Winner Models

| Strategy | Winner task | Selection metric | CV score | Winner hyperparameters |
| --- | ---: | --- | ---: | --- |
| Binary head | 32 | AUROC | 0.983 +/- 0.003 | `conv_channels=384`, `num_conv_layers=3`, `kernel_size=5`, `dropout=0.0`, `learning_rate=0.0001`, `weight_decay=0.0` |
| Multihead/multiclass | 20 | Macro F1 | 0.915 +/- 0.013 | `conv_channels=256`, `num_conv_layers=4`, `kernel_size=5`, `dropout=0.0`, `learning_rate=0.0001`, `weight_decay=0.0` |

## Binary Detection

| Strategy | Binary projection | Inference AUROC | Inference AUPRC | P@P |
| --- | --- | ---: | ---: | ---: |
| Binary head | `positive_class_score` | 0.977 | 0.994 | 0.963 |
| Multihead/multiclass | `one_minus_clean_probability` | 0.986 | 0.996 | 0.968 |

The multihead winner has the stronger clean-vs-backdoor AUROC on the inference set.

## Per-Attack AUROC

| Attack | Binary head AUROC | Multihead AUROC | Better |
| --- | ---: | ---: | --- |
| RIPPLE | 0.988 | 0.996 | Multihead |
| insertsent | 0.967 | 0.981 | Multihead |
| stybkd | 0.988 | 0.994 | Multihead |
| syntactic | 0.967 | 0.972 | Multihead |

## Class Metrics

For the multihead model, accuracy is one-vs-rest accuracy on the full 500-sample inference set, and precision/recall are the direct multiclass argmax metrics. Overall multiclass accuracy is 0.900.

For the binary head, `clean` is measured on the full inference set. Attack rows are measured on each attack plus the shared clean pool using `score >= 0.5`; these rows answer "was this attack detected as backdoored?" rather than "was this attack family named correctly?" Overall binary accuracy at this threshold is 0.946.

| Class | Binary acc. | Binary precision | Binary recall | Multihead acc. | Multihead precision | Multihead recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.946 | 0.929 | 0.790 | 0.948 | 0.870 | 0.870 |
| RIPPLE | 0.895 | 0.826 | 1.000 | 0.986 | 0.951 | 0.980 |
| insertsent | 0.885 | 0.824 | 0.980 | 0.948 | 0.894 | 0.840 |
| stybkd | 0.890 | 0.825 | 0.990 | 0.966 | 0.881 | 0.960 |
| syntactic | 0.880 | 0.822 | 0.970 | 0.952 | 0.904 | 0.850 |

## Takeaway

The multihead CNN is the better threshold-free detector here: it wins on binary AUROC and on every per-attack AUROC. It also gives real attack-family labels, with stronger one-vs-rest accuracy and precision for every attack class. The binary head has very high attack detection recall at the `0.5` threshold, but that comes from a simpler decision: it only needs to decide whether a sample is backdoored, not which attack produced it.
