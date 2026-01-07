# Gradient-Informed Low-Rank Distillation (v2, soft routing)

This is a research prototype for RTE (GLUE) to compare:
- ab_only (LoRA AB only)
- offline_project (train alt only, then offline project/absorb into AB; evaluate AB-only)
- online_nodistill (train AB+alt, no absorb)
- ours_soft_absorb (soft routing using AB vs residual consistency; periodic absorb alt->AB)

Outputs are stored in:
outputs/<model>_<task>/<exp>/<run_tag>/{hpo,final,plots}/

Run:
python scripts/run_rte_suite.py --model_name bert-base-uncased --seeds 2,3 --run_tag s1

Notes:
- Base model is frozen; only adapter parameters are trained.
- Soft routing uses microbatch "votes" (split batch into V chunks) to estimate deltaW-space gradients.
