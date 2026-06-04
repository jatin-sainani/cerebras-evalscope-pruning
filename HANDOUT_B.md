# Handout B — Why this matters and how to use it

**Audience:** developers, test engineers, product, and the customer-facing team. This is not
a simplified Handout A — it answers different questions.

## What this changes for the customer conversation

Today, telling a prospect "yes, this model is good enough for your coding and long-context
workload" means running full benchmark suites on every candidate model — slow and expensive,
so it happens late and rarely. **These pruners let us answer the same go/no-go question on
~10% of the samples.** On the shipped data, a 10%-size LiveCodeBench subset reproduces a
model's full-benchmark score to within ~1 point — close enough to call go/no-go — where naive
"just sample 10% at random" is off by ~6 points and could flip the answer. Concretely: an
eval that took an afternoon and a big inference bill becomes a quick, repeatable check we can
run during a sales cycle, on every new candidate model, without re-spending the full cost.

## How a sales engineer or deployment lead runs it tomorrow

Inside evalscope, it's two evals and one compare — no new tools to learn:

```bash
# 1) full run (the baseline you'd otherwise pay for every time)
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/

# 2) pruned run — 10% of the samples
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "difficulty_stratified", "prune_ratio": 0.1}' \
    --output ./results_pruned/

# 3) did the small run preserve the verdict?
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

`compare_runs` prints, per model, the full vs pruned score, a confidence interval, and a
**GO→GO / NO→NO** verdict check — and flags (with a `*`) any case where the subset is too
small to call the verdict confidently, so you know when to keep more samples. Swap
`live_code_bench` for `aa_lcr` for long-context; use `pruning_strategy: discrimination` when
you're *ranking* several candidate models rather than checking one against a bar.

## What the multimodal probe gives that random sampling cannot

For a customer eyeing multimodal next quarter, the question is narrower than "is the model
smart?" — it's "is its **image encoder** good enough for our images (scanned documents,
charts, high-resolution photos)?" Random sampling from a 12K multimodal benchmark buries the
handful of OCR-heavy, high-resolution, or multi-image cases that actually stress an encoder,
and mixes encoder failures with ordinary reasoning mistakes. The probe **deliberately covers
each visual-stress type** and **isolates the encoder** by comparing each hard-visual question
against a matched easy-visual control: the gap between them is encoder quality, not reasoning
quality. It already runs on the shipped data — and even with one reference model it shows the
encoder is weakest on fine-grained domain imagery (0.60 accuracy) versus everyday photos
(0.82), the kind of axis-level finding random sampling averages away. Random sampling can tell
you a model's average multimodal score; it cannot tell you *why* it failed or whether the
encoder specifically will hold up on the customer's images.

## Why a customer-facing PM should care

It turns "trust us, it's fast and capable" into a number you can put in front of a customer
**this week, for their workload, on every model they ask about** — cheaply enough to redo
when a new model ships. Faster answers, lower eval spend, and a defensible, repeatable
go/no-go instead of a one-off benchmark dump.
