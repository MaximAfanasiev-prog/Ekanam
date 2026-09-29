# YOLO source update 030d697

Reviewed source: ikanam-ai/lct26-street-falcon-reid,
yolo-finetune-metric-learning at 030d697e848f5d8dc0295031679d5c18d60484c3.
Previous integration source: 711294d0033421a5c345e0374acd4d2b478ecc44.

This commit corrects mini-val metric calculation to follow the organizers'
evaluation protocol. It does not introduce a new trained model.
Git blobs are unchanged for the fine-tuned checkpoint, architecture checkpoint,
model.py, final run metadata, and saved final embeddings. ASTs of _rerank_batch
and rerank_sims are identical across the revisions. The retrieve helper refactors
the existing prediction calls without changing the online algorithm.
Consequently no inference-code change or gallery re-encoding is necessary.

The online report now references the current source README and precise candidate
JSON with SHA-256 provenance. Ranking hold-out means remain 66.1% mAP@10,
62.6% Rank-1 and 81.6% Rank-5. The mean F1 for hold-out seeds 7 and 2026 is
0.8858593302218947 (88.6% displayed); TNR is 0.761099305850687.
The README's 0.8841950288121897 F1 averages all THREE seeds, including the tuning
split 42. That aggregate is recorded separately rather than silently replacing
the hold-out card. This is source-reported evaluation, not a newly measured
quality score for the deployed full-train checkpoint or the demo gallery.

## Runtime update and rollback

A new immutable bundle yolo-demo-report-030d697 contains unchanged copies of
checkpoint.pt, base.pt, gallery.npy and ids.json, plus the updated metrics.json
and manifest. Checksums are verified. Model artifact provenance remains 711294d;
report/source review provenance is 030d697. Existing thumbnails are reused.
Gallery remains 11,416 photos. Fixed threshold 0.9402066469192505 and the online
top-k refusal rule are unchanged. Exact-frame exclusion remains disabled.
No UI text removed by the user was reintroduced.

The active maxim-reid-yolo service is operated from the Ekanam checkout using
compose.yolo.yml and the existing external yolo.env, on loopback port 27816.
Windows SSH address stays http://127.0.0.1:8786/.
Rollback configuration: artifacts/street-falcon/yolo-before-030d697.env.
It points to the preceding bundle and the unchanged runtime image.

## Standards postflight and evidence

research-python / Engineering Standards 0.2.9; no registered exceptions.
Runtime-only report update; source data mounted read-only during checks.
Three real queries across train/query/gallery: candidate and active responses
match exactly for accepted, threshold, matches, model_version and decision_score.
30 result thumbnails decoded; all bundle checksums, shapes and norms verified.
Chromium existing smoke passed: model cards, photo results, zero matches,
recovery, errors and mobile layout. Tests were targeted to this report-only
change; model retraining and a new full-dataset quality evaluation were not run.
Diff reviewed: aggregate report and documentation only, no model weights,
organizer images, per-image results or credentials added to Git.
No GitLab policy-gate, public deployment or production-acceptance claim.
