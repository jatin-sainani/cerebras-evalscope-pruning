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

## Part B — a forward-looking MMMU image-encoder probe

Goal: through the **standard OpenAI chat (`image_url`) interface only**, tell whether a
candidate's *vision encoder* (not its text reasoning) is good enough. Generic MMMU accuracy
conflates the two. The design (reusing the same machinery; `mmmu_pruned` is wired, the probe
selector is specified in `evalscope_ext/tools/` design notes):

1. **Tag all ~12K `MMMU/MMMU` items along encoder-stress axes**, cheaply and model-free:
   native resolution / downscale loss, **text-in-image / OCR load** (OCR token count),
   fine-grained detail & counting, multi-image cross-binding, and chart/diagram density.
2. **Select an axis-balanced probe** with the stratified+diversity strategies so every stress
   axis is covered (random sampling drowns these — OCR-heavy or multi-image items are a small
   fraction of 12K and would barely appear).
3. **Measure the encoder, not the reasoner, by paired contrast:** for each hard-visual item,
   pair it with a control that needs the *same* reasoning but minimal visual fidelity (e.g.
   the same question with the key fact given as text, or a clean high-res render). The
   per-axis **accuracy gap (hard − control)** isolates encoder degradation: a model that
   collapses on OCR/high-res items but holds on controls has a weak encoder, independent of
   its reasoning ability. Report per-axis gaps + an aggregate encoder-quality index.

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
