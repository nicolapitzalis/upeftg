# Spectral Feature Extraction

This document explains the active spectral feature extraction subsystem under
`upeftguard.features`. It covers its purpose, mathematical model, command-line
configuration, source layout, execution flow, and output contract.

## Purpose

The spectral extractor converts each PEFT adapter into a fixed numerical
description that can be consumed by the supervised backdoor detectors.

For a LoRA block with factors

```text
A: [rank, input_dimension]
B: [output_dimension, rank]
```

the effective weight update is

```text
delta_W = B @ A
```

The extractor describes the magnitude and spectral shape of `delta_W` without
loading the corresponding base model. It operates directly on
`adapter_model.safetensors` files referenced by a dataset manifest.

Its main goals are:

1. Compute informative features from LoRA weight updates without reconstructing
   or running the base language model.
2. Give every adapter in an extraction run the same deterministic row and
   column schema.
3. Support multiple attention naming/layout conventions while making
   incompatible schemas fail explicitly or run in separate extraction groups.
4. Preserve enough metadata and provenance for aggregation, training,
   inference, and older artifact readers.

The primary public function is
`upeftguard.features.spectral.extract_spectral_features`. It returns:

```python
features, labels, model_names, metadata
```

- `features` is a `float32` array with shape
  `[number_of_kept_adapters, feature_dimension]`.
- `labels` is an `int32` vector when every kept manifest item has a label;
  otherwise it is `None`.
- `model_names` identifies the feature rows.
- `metadata` describes the extractor configuration, feature names, block
  layout, LoRA dimensions, schema, and skipped adapters.

The spectral package computes the in-memory result. The feature registry and
artifact packages are responsible for saving it and its companion files.

## Efficient spectral computation

Most features depend on the singular values of `delta_W`. Computing an SVD of
the full `[output_dimension, input_dimension]` matrix would be unnecessarily
expensive because a LoRA update is low rank.

The implementation instead computes reduced QR decompositions:

```text
B   = Q_b R_b
A^T = Q_a R_a
```

The nonzero singular values of `B @ A` are the singular values of the much
smaller matrix

```text
R_b @ R_a^T
```

whose dimensions are controlled by the LoRA rank. This is implemented in
`upeftguard/features/delta.py`.

Entrywise statistics are different: they require the values of `B @ A`, not
only its spectrum. Those values are either materialized as one dense matrix or
calculated in streamed blocks by `upeftguard/features/norms.py`.

## Source layout

### Adjacent feature modules

`upeftguard/features/adapters.py`

: Provides low-level safetensors inspection and chunked tensor/matrix
  iteration utilities.

`upeftguard/features/delta.py`

: Discovers matching LoRA A/B factors, supports AdaLoRA E scaling, validates
  schemas, loads effective factors, and computes low-rank spectra.

`upeftguard/features/norms.py`

: Computes dense or streaming entrywise moments and decides whether `auto`
  mode can safely materialize an effective delta.

`upeftguard/features/registry.py`

: Selects the registered extractor, invokes spectral extraction, writes the
  feature array and companion artifacts, and records warnings.

### `upeftguard.features.spectral`

`spectral/__init__.py`

: Defines the public import surface by exporting selected names directly from
  the decomposed modules. It contains no extraction algorithm.

`spectral/config.py`

: Owns extractor defaults, supported feature groups, validation, moment-source
  expansion, q+v modes, attention granularities, and the normalized parameter
  payload stored in metadata.

`spectral/layout.py`

: Interprets raw adapter block names, pairs q and v projections, infers
  supported attention-head layouts, slices head factors, and constructs
  deterministic block and feature names.

`spectral/computation.py`

: Computes the requested numerical values for one LoRA block or attention
  head. It avoids calculating singular values or entrywise moments when the
  requested feature set does not require them.

`spectral/metadata.py`

: Normalizes metadata, derives block and layer identities, reconstructs LoRA
  dimension mappings from old and new artifacts, summarizes schemas, and
  removes internal merge-only fields from public metadata.

`spectral/extraction.py`

: Orchestrates extraction across all manifest items: schema selection and
  validation, block loading, module/head handling, q+v construction, feature
  computation, recoverable adapter skipping, feature-name validation, and
  metadata assembly.

The majority of the code handles schemas, layouts, deterministic naming, and
metadata compatibility. The individual numerical formulas are a smaller part
of the subsystem.

## Command-line flags

The flags are registered in `upeftguard/commands/common.py`. Extraction is the
stage that computes new spectral values. Aggregation and training may accept a
subset of the same options to select columns or validate the feature artifact's
recorded configuration; they do not recompute adapter spectra.

### `--features`

Selects one or more feature groups. The default set is:

```text
energy
kurtosis
l1_norm
l2_norm
linf_norm
mean_abs
concentration_of_energy
sv_topk
stable_rank
spectral_entropy
effective_rank
```

All supported values are:

| Feature | Implementation meaning |
| --- | --- |
| `block_rank` | Effective rank scale used by normalized features. |
| `energy` | Squared Frobenius norm, `sum(s_i ** 2)`. |
| `energy_per_rank` | Energy divided by block rank. |
| `l2_norm` | Frobenius norm, `sqrt(energy)`. |
| `l2_norm_per_sqrt_rank` | L2 norm divided by the square root of block rank. |
| `concentration_of_energy` | Largest singular value divided by the sum of singular values. |
| `sv_topk` | First `k` singular values, padded with zeros when needed. |
| `stable_rank` | Energy divided by the squared largest singular value. |
| `stable_rank_frac` | Stable rank divided by block rank. |
| `spectral_entropy` | Entropy of the normalized singular-value distribution. |
| `normalized_spectral_entropy` | Spectral entropy divided by log-rank. |
| `effective_rank` | Exponential of spectral entropy. |
| `effective_rank_frac` | Effective rank divided by block rank. |
| `kurtosis` | Kurtosis of delta entries or singular values, depending on the moment source. |
| `l1_norm` | L1 norm of delta entries or singular values, depending on the moment source. |
| `l1_norm_per_rank` | Entrywise L1 norm divided by block rank. |
| `sv_l1_norm_per_rank` | Singular-value L1 norm divided by block rank. |
| `linf_norm` | Maximum absolute delta entry or singular value. |
| `mean_abs` | Mean absolute delta entry or singular value. |
| `mean_abs_per_rank` | Mean absolute value divided by block rank. |

### `--spectral-moment-source {entrywise,sv,both}`

Default: `sv`.

This controls the source of moment-like feature groups:

```text
kurtosis
l1_norm
l1_norm_per_rank
linf_norm
mean_abs
mean_abs_per_rank
```

- `entrywise` computes them over the entries of `delta_W`.
- `sv` computes them over the singular values of `delta_W`.
- `both` emits both forms.

For example, requesting `kurtosis` with `entrywise` produces a column named
`<block>.kurtosis`. With `sv`, it produces `<block>.sv_kurtosis`. With `both`,
both columns are emitted.

This option changes the scientific meaning of the selected features. In
particular, the entrywise L1 norm and the sum of singular values are not the
same matrix quantity.

### `--spectral-sv-top-k K`

Default: `8`.

Controls the number of columns emitted for `sv_topk`:

```text
<block>.sv_1
<block>.sv_2
...
<block>.sv_K
```

The value must be positive. It has no effect when `sv_topk` is not selected.

### `--spectral-qv-sum-mode {none,append,only}`

Controls the treatment of attention q and v projections:

- `none` emits the original q and v blocks.
- `append` emits q, v, and an additional q+v block.
- `only` emits only q+v blocks.

For compatible q/v factors, the derived update is

```text
delta_W_qv = delta_W_q + delta_W_v
```

without constructing the sum directly. The implementation uses

```text
A_qv = concatenate([A_q, A_v], axis=0)
B_qv = concatenate([B_q, B_v], axis=1)
```

so that

```text
B_qv @ A_qv = B_q @ A_q + B_v @ A_v
```

This must happen before spectral features are calculated. Nonlinear spectral
features of q+v cannot be reconstructed later by adding the q and v feature
values.

The extraction/full default is `none`. Aggregation and training paths can use
different command-specific defaults, so a full experiment explicitly
propagates one selected value through all stages.

### `--spectral-attention-granularity {module,head}`

Default: `module`.

- `module` computes one block of features for each complete LoRA projection.
- `head` slices the output rows of the B factor and computes one block per
  attention head.

Module feature names resemble:

```text
layer0.self_attn.q_proj.energy
layer0.self_attn.v_proj.energy
```

Head-level names resemble:

```text
layer0.self_attn.q_proj.head00.energy
layer0.self_attn.v_proj.head00.energy
layer0.self_attn.qv_sum.head00.energy
```

The currently recognized layouts are:

| Architecture convention | Projection names | Head dimension |
| --- | --- | ---: |
| LLaMA/Qwen | `q_proj`, `v_proj` | 128 |
| T5 | `q`, `v` | 64 |
| RoBERTa/BERT | `query`, `value` | 64 |

Head mode is intentionally strict. Every included base block must be a
recognized q/v attention projection. Unknown layouts, fused QKV projections,
MLP blocks, non-divisible output dimensions, and mismatched q/v head layouts
produce explicit errors.

### `--spectral-entrywise-delta-mode {auto,dense,stream}`

Default: `auto`.

This option applies only when entrywise moments are requested:

- `dense` materializes `B @ A`.
- `stream` calculates delta entries in bounded blocks.
- `auto` estimates the dense working set and chooses between the two.

With the default `--spectral-moment-source sv`, no entrywise moments are needed,
so this option generally does not affect extraction. Metadata records how many
blocks actually used dense and streaming execution.

### `--stream-block-size N`

Default: `131072`.

Controls the approximate number of delta entries processed per streaming
block. It only matters when entrywise moments use streaming. Smaller values
reduce peak memory; larger values reduce iteration overhead.

The value must be positive.

### `--dtype {float32,float64}`

Default: `float32`.

Controls the dtype used to load factors and perform intermediate numerical
work. The final feature matrix is currently converted to `float32` regardless
of this setting. Consequently, `float64` provides more intermediate precision
but does not create a `float64` output artifact.

## Flag interactions

Several flags are meaningful only in combination:

- `--spectral-sv-top-k` only matters when `sv_topk` is selected.
- `--spectral-entrywise-delta-mode` and `--stream-block-size` only matter when
  the resolved features require entrywise moments.
- `--spectral-qv-sum-mode append|only` requires complete compatible q/v pairs.
- `--spectral-attention-granularity head` requires a recognized attention
  naming convention and output dimension.
- Head-level q+v additionally requires q and v to have identical inferred head
  layouts.
- Selecting only `block_rank` avoids both SVD and entrywise delta computation.

Examples:

```bash
# Default module-level spectrum
python -m upeftguard.cli experiment extract \
  --manifest-json manifests/single_datasets/llama2_7b_toxic_backdoors_hard.json

# Singular values only
python -m upeftguard.cli experiment extract \
  --manifest-json <MANIFEST> \
  --features sv_topk \
  --spectral-sv-top-k 16

# Entrywise moments using bounded memory
python -m upeftguard.cli experiment extract \
  --manifest-json <MANIFEST> \
  --features kurtosis l1_norm linf_norm mean_abs \
  --spectral-moment-source entrywise \
  --spectral-entrywise-delta-mode stream \
  --stream-block-size 65536

# Separate q/v heads plus derived q+v heads
python -m upeftguard.cli experiment extract \
  --manifest-json <MANIFEST> \
  --spectral-qv-sum-mode append \
  --spectral-attention-granularity head
```

## Extraction flow

`extract_spectral_features` performs the following sequence:

1. Validate that adapters were supplied and numerical parameters are positive.
2. Resolve and deduplicate requested feature groups.
3. Expand moment features according to the selected moment source.
4. Find the first readable adapter and use it as the reference delta schema.
5. Decide whether original blocks, q+v blocks, or both are required.
6. Pair q and v modules and infer head layouts where requested.
7. Determine whether the requested features need singular values, top-k values,
   spectral scalars, entrywise moments, or singular-value moments.
8. For each manifest item:
   - open its safetensors file;
   - validate it against the reference schema;
   - load effective A/B factors, including AdaLoRA scaling;
   - compute module-level or head-level features;
   - retain q/v factors and calculate derived q+v features when requested;
   - record and skip recoverable read or schema failures.
9. Generate feature names in exactly the same order as the numerical row.
10. Verify that the matrix width equals the number of generated feature names.
11. Build and sanitize schema, dimension, layout, and execution metadata.
12. Return the feature matrix and aligned row information.

## Schema behavior and skipped adapters

The first readable adapter supplies the expected tensor pairs and shapes.
Ordinary LoRA extraction expects subsequent adapters to match that schema.
AdaLoRA schemas can vary in rank when their input/output dimensions and scaling
tensors remain compatible.

Slurm extraction handles broader heterogeneous manifests before this function
is called: `upeftguard.orchestration.sharding.prepare_schema_sharded_manifests`
groups adapters by compatible schema, after which each group is extracted
separately.

The extractor treats `OSError`, `SafetensorError`, and `ValueError` while
reading or validating an individual adapter as recoverable item failures. Such
adapters are omitted from the feature matrix and recorded in metadata:

```json
{
  "skipped_model_count": 1,
  "skipped_models": [
    {
      "model_name": "...",
      "adapter_path": ".../adapter_model.safetensors",
      "label": 1,
      "exception_type": "SafetensorError",
      "exception_message": "..."
    }
  ]
}
```

If no adapter is readable, extraction fails rather than producing an empty
artifact.

## Output naming and dimensionality

Each output column combines a block name and emitted feature name:

```text
<block>.<feature>
```

Top singular values expand to multiple columns:

```text
<block>.sv_1
<block>.sv_2
...
```

At module granularity, the approximate feature dimension is

```text
number_of_selected_blocks * emitted_features_per_block
```

where `sv_topk` contributes `K` columns rather than one. Head granularity
multiplies the relevant block count by the number of heads. `qv_sum_mode=append`
adds one derived q+v block for every compatible q/v attention pair.

The resulting artifact is normally saved with companion files such as:

```text
spectral_features.npy
spectral_labels.npy
spectral_model_names.json
spectral_metadata.json
dataset_reference_report.json
.spectral_metadata_state.json
.dataset_reference_state.json
```

The exact filenames can vary between a direct extraction run and a merged
artifact, so consumers resolve them through the artifact path utilities.

## Metadata contract

Important metadata fields include:

| Field | Purpose |
| --- | --- |
| `extractor` | Identifies this as a spectral artifact. |
| `extractor_version` | Version of the spectral extraction contract. |
| `delta_schema_version` | Version of the LoRA delta schema. |
| `input_n_models` | Manifest items presented to extraction. |
| `n_models` | Successfully extracted rows. |
| `resolved_features` | User-facing feature groups selected. |
| `feature_names` | Ordered semantic name of every matrix column. |
| `feature_dim` | Number of matrix columns. |
| `block_names` | Ordered blocks represented by the columns. |
| `base_block_names` | Original LoRA blocks before q+v selection. |
| `qv_sum_block_names` | Derived q+v blocks. |
| `spectral_qv_sum_mode` | Resolved q+v configuration. |
| `spectral_moment_source` | Resolved entrywise/SV moment source. |
| `spectral_entrywise_delta_mode` | Requested delta materialization mode. |
| `spectral_attention_granularity` | Module or head layout. |
| `entrywise_dense_block_count` | Blocks whose moments used dense deltas. |
| `entrywise_stream_block_count` | Blocks whose moments used streamed deltas. |
| `sv_top_k` | Number of top singular-value columns per block. |
| `schema_layout_summary` | Layer count and adapter-dimension summary. |
| `attention_head_layouts` | Per-block head counts and dimensions in head mode. |
| `skipped_models` | Recoverable adapter failures, when any occur. |

Downstream aggregation relies on this metadata to map architecture-specific
block names to canonical depth and slot axes. Training relies on it to ensure
that the selected model receives a compatible representation.

## Boundary with downstream stages

The spectral extractor emits a flat, semantically named feature row for each
adapter. It does not train a detector and does not create the final CNN or
Transformer input layout.

The next stage, `upeftguard.artifacts.aggregation`, uses block names and metadata
to transform the flat representation into an architecture-independent tensor:

```text
[sample, depth, slot, emitted_feature]
```

with masks for missing depths, slots, and values. That structured artifact is
then consumed by the CNN, DANN, and Transformer supervised backends. CNN/DANN
input vectors include value-validity channels. Transformer input likewise
concatenates one observed/missing channel per emitted feature, so a missing
zero is distinguishable from a genuine normalized zero.
