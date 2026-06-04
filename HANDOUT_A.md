# Handout A — Why this works (technical)

**Audience:** an engineer who could have built this. **Extension:** `evalscope_ext/`,
developed against evalscope `@ c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d`.

## The problem I set out to solve

A sales/deployment team needs a fast *"is this model good enough?"* answer for **code
generation** (LiveCodeBench v5) and **long-context reasoning** (AA-LCR). Running the full
suites on every candidate is expensive, and most of that cost is wasted: on the shipped
3-model panel, **only 111/315 LiveCodeBench items (35%) and 43/100 AA-LCR items (43%)
produce *any* disagreement between models.** The rest are unanimous pass/fail — they cost
inference but move no decision.

So pruning is not "sample fewer rows"; it is *"keep the rows that carry decision signal,
in the right proportions, for a model we have not seen yet."* The hard constraint is the
last clause: the subset is chosen **only from reference-model behaviour** (the panel's
per-sample scores), never from the model under test, so it cannot overfit to any one model.
I validate that directly with **leave-one-model-out**: build the subset from two models,
then check how well it predicts the *third, held-out* model's full-benchmark score.

## What I built

A framework-agnostic pruning core (`evalscope_ext/pruning/`) plugged into evalscope via a
`PruningMixin` that overrides `load_dataset` — pruning is a global selection over the loaded
subset, so a per-sample filter is the wrong layer. Pruned variants `live_code_bench_pruned`,
`aa_lcr_pruned`, `mmmu_pruned` register as ordinary benchmarks; the strategy and ratio come
from `--dataset-args`. Five registered strategies, none a forbidden baseline:

- **`difficulty_stratified`** — bin by difficulty (fraction of the panel that fails), allocate
  the budget across bins *proportionally to bin mass* (largest-remainder), pick within a bin
  by farthest-first coverage. Distribution-preserving ⇒ the pruned mean is an **unbiased**
  estimator of the true score.
- **`discrimination`** — keep items the panel *splits*, scored by Fisher information `p(1−p)`
  × point-biserial alignment with model ability (an IRT-flavoured item-information criterion),
  coverage-diversified so the kept splitters aren't redundant. This is a *ranking* tool.
- **`diversity`** — k-center coverage over item features (behavioural fingerprint, or text
  embedding when present).
- **`judge_noise_aware`** — for the LLM-judged AA-LCR: stratified coverage that prefers
  verdict-stable items; a no-op on deterministic graders.
- **`balanced`** (default) — stratified backbone (unbiased) with within-stratum discrimination
  (power), and for judged benchmarks it first drops the noisiest 15% by judge-variance proxy.

## How much I pruned, and why the subset is sufficient

Headline: **keep 10%** (LCB 32/315, AA-LCR 10/100). Leave-one-model-out, error = |pruned −
full| score for the held-out model, averaged over the 3 held-out choices:

| benchmark | strategy | MAE @10% | vs uniform-random | top-k-hardest (forbidden) |
|---|---|---|---|---|
| LiveCodeBench | `difficulty_stratified` | **0.009** | random 0.060 → **86% lower** | 0.421 |
| LiveCodeBench | `balanced` | 0.040 | 34% lower | — |
| AA-LCR | `balanced` | 0.073 | random 0.124 → 41% lower | 0.260 |
| AA-LCR | `discrimination` | 0.060 | 52% lower | — |

A 10% LCB subset pins a held-out model's true score to ~1 point — comfortably inside any
sane go/no-go margin — while uniform random is off by ~6 and the forbidden top-k-hardest is
off by ~42 (it only ever sees impossible items, so every model "fails"). The advantage is
largest at aggressive ratios (5–10%) and narrows by ~20–30% kept, exactly as expected: when
you keep more, even random becomes unbiased, so pruning intelligence matters most where it
saves the most. **Recommendation:** `difficulty_stratified` for an absolute go/no-go bar
(unbiased estimate); `discrimination` when the question is *ranking* candidates; `balanced`
when you want one robust default.

## Part B — MMMU image-encoder probe (working code)

Goal: through the **standard OpenAI chat (`image_url`) interface only**, tell whether a
candidate's *vision encoder* (not its text reasoning) is good enough. Generic MMMU accuracy
conflates the two. Implemented in `evalscope_ext/tools/mmmu_probe.py` + the registered
`mmmu_encoder_probe` strategy — it runs on the shipped data offline and against a live
endpoint:

1. **Tag every item by encoder-stress axis** from MMMU's own `img_type` field (no model
   needed): *text-in-image* (Tables, Screenshots, …), *chart/diagram* (Diagrams, Plots,
   Blueprints), *fine-grained* (Medical, Microscopic, Pathological, Chemical), *multi-image*
   (image count > 1), *natural* (Photographs, Paintings). The 30+ MMMU categories map onto
   these five axes.
2. **Select an axis-balanced, difficulty-spread probe** with the `mmmu_encoder_probe`
   strategy (same registry/mixin as Part A). Uniform sampling of 12K buries the stressing
   categories — fine-grained and multi-image are a small slice; balancing guarantees each
   failure mode is probed. Runs in-framework: `--datasets mmmu_pruned
   --dataset-args '{"pruning_strategy":"mmmu_encoder_probe","prune_ratio":0.05}'`.
3. **Measure the encoder by paired contrast** (`measure_encoder`): each hard-visual item is
   asked twice through `image_url` — once with the real image, once with a text-only control
   that conveys the same reasoning content (the MMMU explanation) with no image. The per-axis
   **gap = acc(control) − acc(hard)** isolates encoder degradation; the aggregate
   **encoder-quality index = 1 − mean gap**.

**Already-visible signal (offline, shipped reference model, no live calls):** per-axis
accuracy is **fine-grained 0.60 · multi-image 0.68 · chart/diagram 0.68 · text-in-image 0.72
· natural 0.82** — the encoder is weakest on fine-grained domain imagery, **22 points** below
natural images. That ordering is the encoder-stress signal the probe is built to expose, and
uniform sampling (weighted to the common chart/natural categories) would wash it out. The
live paired contrast turns each axis into a clean hard-vs-control gap for a candidate model.

## Assumptions

- The offline signal index aligns with evalscope's loaded sample order (both derive from the
  upstream dataset's canonical order); the mixin `reindex()`es and warns if signal coverage
  of a loaded subset drops below 50%.
- A 3-model panel is a usable but small proxy for "model space"; difficulty/discrimination
  estimates are coarse (Fisher info, not a fully-fit IRT, which is unstable at 3 models).
- AA-LCR judge variance is approximated from single-pass panel dispersion `p(1−p)` — a proxy
  for verdict fragility, not a measured judge variance (see below).

## What would change with…

- **(a) more data / a larger model panel:** fit a proper 2-PL IRT (stable item discrimination
  & difficulty), and learn per-axis difficulty for MMMU rather than heuristic tags.
- **(b) a live endpoint:** measure AA-LCR judge variance *directly* by re-judging k times and
  pruning toward low-variance items (replacing the proxy); run the MMMU paired-contrast probe
  for real and calibrate the go/no-go threshold against measured customer outcomes.
- **(c) more time:** text/code embeddings for true semantic-diversity pruning; a cost-aware
  objective (minimise subset size s.t. the verdict CI stays off the threshold); and an
  upstream-PR entry-point hook so `evalscope_ext` is auto-discovered without an explicit import.
