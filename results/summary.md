# StaleRAG++ experiment summary

**Simulated run.** Readers in this run are rule-based simulations with hand-set parameters and the NLI scorer is a lexical heuristic on synthetic data. These numbers verify that the pipeline works end to end. They are not measurements of Llama-3.1-8B, Qwen2.5-7B or Mistral-7B and must not be quoted as project results.

- data source: synthetic; examples: 2000 (train 1000, val 400, test 600, split by entity)
- seeds: [0, 1, 2]; bootstrap resamples: 1000; coverage target: 0.8
- detector backend: sklearn_hist_gradient_boosting; NLI scorer: {'kind': 'lexical'}
- wall time (s): {'data': 0.1, 'features': 0.3, 'detector': 16.1, 'readers': 3.0, 'total': 24.7}

## Evidence reliability detector (positive class = stale)

| metric | value [95% bootstrap CI] |
|---|---|
| AUROC | 0.986 [0.981, 0.990] |
| F1 | 0.924 [0.908, 0.939] |
| AUPRC | 0.986 [0.981, 0.990] |

AUROC across seeds: 0.986 +/- 0.0001

### Feature ablation

| variant | AUROC | F1 | AUPRC |
|---|---|---|---|
| without_temporal | 0.968 [0.960, 0.976] | 0.892 [0.874, 0.912] | 0.969 [0.960, 0.976] |
| only_temporal | 0.934 [0.920, 0.945] | 0.851 [0.830, 0.872] | 0.930 [0.916, 0.944] |
| without_nli | 0.986 [0.982, 0.990] | 0.925 [0.909, 0.939] | 0.986 [0.981, 0.990] |
| only_nli | 0.780 [0.756, 0.804] | 0.697 [0.667, 0.728] | 0.746 [0.717, 0.777] |
| without_lexical | 0.962 [0.953, 0.970] | 0.885 [0.865, 0.905] | 0.963 [0.954, 0.971] |
| only_lexical | 0.943 [0.932, 0.954] | 0.832 [0.807, 0.855] | 0.944 [0.932, 0.955] |
| full | 0.986 [0.981, 0.990] | 0.924 [0.908, 0.939] | 0.986 [0.981, 0.990] |

### Generalization

| setting | AUROC | F1 |
|---|---|---|
| unseen entity (main split) | 0.986 [0.981, 0.990] | 0.924 [0.908, 0.939] |
| random split | 0.989 [0.985, 0.992] | 0.928 [0.913, 0.943] |
| held-out property: chief executive officer | 0.953 [0.941, 0.965] | 0.832 [0.805, 0.857] |
| held-out property: head coach | 0.986 [0.981, 0.991] | 0.920 [0.899, 0.939] |
| held-out property: mayor | 0.915 [0.895, 0.930] | 0.782 [0.749, 0.811] |
| held-out property: president | 0.950 [0.937, 0.962] | 0.841 [0.811, 0.868] |
| held-out property: prime minister | 0.984 [0.978, 0.989] | 0.926 [0.907, 0.944] |

## Reader: sim_order_biased (simulated)

### Retrieval conditions

| condition | stale rate | current accuracy | other |
|---|---|---|---|
| closed_book | 0.504 [0.482, 0.526] | 0.183 [0.165, 0.201] | 0.313 [0.293, 0.333] |
| stale_only | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| current_only | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mixed_stale_first | 0.683 [0.659, 0.704] | 0.317 [0.296, 0.341] | 0.000 [0.000, 0.000] |
| mixed_current_first | 0.235 [0.217, 0.256] | 0.765 [0.744, 0.783] | 0.000 [0.000, 0.000] |
| vanilla | 0.504 [0.475, 0.534] | 0.496 [0.466, 0.525] | 0.000 [0.000, 0.000] |
| rerank_sf | 0.252 [0.232, 0.276] | 0.748 [0.724, 0.768] | 0.000 [0.000, 0.000] |
| filter_sf | 0.059 [0.042, 0.076] | 0.941 [0.924, 0.958] | 0.000 [0.000, 0.000] |

### Passage-order sensitivity (answer flip rate, stale-first vs current-first)

| method | flip rate | accuracy gap | reduction vs vanilla |
|---|---|---|---|
| vanilla | 0.597 [0.573, 0.618] | 0.448 | - |
| rerank | 0.000 [0.000, 0.000] | 0.000 | 1.000 [1.000, 1.000] |
| filter | 0.002 [0.000, 0.005] | 0.002 | 0.997 [0.992, 1.000] |

### Selective prediction

| method | coverage | accuracy on answered | stale rate (all questions) | accuracy w/o abstention | AURC |
|---|---|---|---|---|---|
| vanilla | 0.810 [0.790, 0.828] | 0.495 [0.466, 0.525] | 0.409 [0.383, 0.435] | 0.496 | 0.492 |
| rerank | 0.802 [0.782, 0.819] | 0.924 [0.904, 0.941] | 0.061 [0.047, 0.077] | 0.748 | 0.047 |
| filter | 0.791 [0.761, 0.821] | 0.978 [0.964, 0.990] | 0.017 [0.008, 0.029] | 0.941 | 0.014 |

### Relative stale-rate reduction vs vanilla RAG (no abstention)

| comparison | reduction [95% CI] |
|---|---|
| rerank_no_abstain_vs_vanilla | 0.500 [0.455, 0.540] |
| rerank_abstain_vs_vanilla | 0.879 [0.847, 0.907] |
| filter_no_abstain_vs_vanilla | 0.883 [0.849, 0.916] |
| filter_abstain_vs_vanilla | 0.966 [0.943, 0.984] |
| vanilla_abstain_vs_vanilla | 0.189 [0.163, 0.217] |

## Reader: sim_recency_aware (simulated)

### Retrieval conditions

| condition | stale rate | current accuracy | other |
|---|---|---|---|
| closed_book | 0.356 [0.335, 0.377] | 0.342 [0.321, 0.364] | 0.302 [0.280, 0.323] |
| stale_only | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| current_only | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mixed_stale_first | 0.383 [0.358, 0.409] | 0.617 [0.591, 0.642] | 0.000 [0.000, 0.000] |
| mixed_current_first | 0.394 [0.371, 0.419] | 0.606 [0.581, 0.629] | 0.000 [0.000, 0.000] |
| vanilla | 0.395 [0.369, 0.422] | 0.605 [0.578, 0.631] | 0.000 [0.000, 0.000] |
| rerank_sf | 0.399 [0.376, 0.424] | 0.601 [0.576, 0.624] | 0.000 [0.000, 0.000] |
| filter_sf | 0.066 [0.048, 0.083] | 0.934 [0.917, 0.952] | 0.000 [0.000, 0.000] |

### Passage-order sensitivity (answer flip rate, stale-first vs current-first)

| method | flip rate | accuracy gap | reduction vs vanilla |
|---|---|---|---|
| vanilla | 0.429 [0.407, 0.454] | 0.019 | - |
| rerank | 0.000 [0.000, 0.000] | 0.000 | 1.000 [1.000, 1.000] |
| filter | 0.002 [0.000, 0.005] | 0.002 | 0.996 [0.988, 1.000] |

### Selective prediction

| method | coverage | accuracy on answered | stale rate (all questions) | accuracy w/o abstention | AURC |
|---|---|---|---|---|---|
| vanilla | 0.829 [0.812, 0.846] | 0.602 [0.574, 0.629] | 0.331 [0.307, 0.356] | 0.605 | 0.398 |
| rerank | 0.782 [0.759, 0.803] | 0.769 [0.741, 0.793] | 0.181 [0.159, 0.205] | 0.601 | 0.106 |
| filter | 0.800 [0.767, 0.829] | 0.979 [0.965, 0.990] | 0.017 [0.008, 0.028] | 0.934 | 0.013 |

### Relative stale-rate reduction vs vanilla RAG (no abstention)

| comparison | reduction [95% CI] |
|---|---|
| rerank_no_abstain_vs_vanilla | -0.010 [-0.069, 0.047] |
| rerank_abstain_vs_vanilla | 0.541 [0.487, 0.595] |
| filter_no_abstain_vs_vanilla | 0.834 [0.788, 0.877] |
| filter_abstain_vs_vanilla | 0.958 [0.929, 0.980] |
| vanilla_abstain_vs_vanilla | 0.163 [0.138, 0.192] |

## Reader: sim_stale_memory (simulated)

### Retrieval conditions

| condition | stale rate | current accuracy | other |
|---|---|---|---|
| closed_book | 0.696 [0.674, 0.716] | 0.103 [0.088, 0.117] | 0.202 [0.184, 0.221] |
| stale_only | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| current_only | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mixed_stale_first | 0.574 [0.552, 0.597] | 0.426 [0.403, 0.448] | 0.000 [0.000, 0.000] |
| mixed_current_first | 0.382 [0.358, 0.404] | 0.618 [0.596, 0.642] | 0.000 [0.000, 0.000] |
| vanilla | 0.496 [0.472, 0.519] | 0.504 [0.481, 0.528] | 0.000 [0.000, 0.000] |
| rerank_sf | 0.395 [0.372, 0.418] | 0.605 [0.582, 0.628] | 0.000 [0.000, 0.000] |
| filter_sf | 0.061 [0.044, 0.079] | 0.939 [0.921, 0.956] | 0.000 [0.000, 0.000] |

### Passage-order sensitivity (answer flip rate, stale-first vs current-first)

| method | flip rate | accuracy gap | reduction vs vanilla |
|---|---|---|---|
| vanilla | 0.518 [0.496, 0.541] | 0.193 | - |
| rerank | 0.000 [0.000, 0.000] | 0.000 | 1.000 [1.000, 1.000] |
| filter | 0.002 [0.000, 0.005] | 0.002 | 0.997 [0.990, 1.000] |

### Selective prediction

| method | coverage | accuracy on answered | stale rate (all questions) | accuracy w/o abstention | AURC |
|---|---|---|---|---|---|
| vanilla | 0.782 [0.764, 0.801] | 0.511 [0.484, 0.537] | 0.382 [0.359, 0.404] | 0.504 | 0.493 |
| rerank | 0.799 [0.777, 0.821] | 0.758 [0.730, 0.782] | 0.194 [0.173, 0.218] | 0.605 | 0.108 |
| filter | 0.806 [0.775, 0.835] | 0.980 [0.966, 0.991] | 0.016 [0.007, 0.027] | 0.939 | 0.014 |

### Relative stale-rate reduction vs vanilla RAG (no abstention)

| comparison | reduction [95% CI] |
|---|---|
| rerank_no_abstain_vs_vanilla | 0.203 [0.155, 0.246] |
| rerank_abstain_vs_vanilla | 0.609 [0.558, 0.652] |
| filter_no_abstain_vs_vanilla | 0.877 [0.840, 0.912] |
| filter_abstain_vs_vanilla | 0.967 [0.946, 0.985] |
| vanilla_abstain_vs_vanilla | 0.229 [0.201, 0.255] |
