# DockMut-1 and the layout of the archive

This file documents the archive in detail. The overview of the study is README.md.
DockMut-1 is the preceding analysis of the same project: re-docking of 57 DOCKSTRING receptors and their in silico alanine truncations with Uni-Dock 1.2.0 (305,227 valid scores),
the surrogates trained on them (pocket-edit test), the evaluation outputs and all analysis code of the manuscript
"Jev-Like Shared-State Networks for Fast Structure-Based Screening"
(H. Kong, Seoul National University, hskong@snu.ac.kr). The manuscript reports the cost and the known-pocket recall of Jev-like networks;
the DockMut-1 pocket-edit analyses are in its Supporting Information.

The registered analysis plan (code/PREREG.md) and its hash ledger (code/prereg_ledger.json) were written before the
held-out evaluation of any interventional model. Addendum 5 of the plan was written after the registered outcomes were
known and declares the correction of two permutation controls, the additional seeds and the post hoc analyses. Addendum 6 restates
the screening comparison with the hit-probability head of the baselines. Addendum 7 was written before the six hit-head networks
were trained. Addendum 8 was written before the first run of the joint networks (cross-attention and pooled concatenation) and fixes the
conditions, the outcomes and the margin of -0.03 for the claim of equal recall. Addendum 9 was written before the cross-validation over target families and the full-scale timing, addendum 10 after most of
these experiments (the RTX 4090 timing followed it) and before the known-pocket evaluation of the existing checkpoints (the manuscript is restricted to known pockets), and addendum 11 after that evaluation and before the cross-fitted
known-pocket evaluation; all outcomes of these experiments are in this archive. The docking rate benchmark, the throughput of the three network families, the budget summary (eval/wt_budget_summary.json)
and the ML-guided docking baseline (eval/ml_guided_dev.json) were added after the registered outcomes and are post hoc; the
Supporting Information discloses them.

## Layout
- results/<TARGET>.csv: docking score of every receptor variant for every ligand of that target. Columns include the
  receptor id (`wt` for wild-type replicates, `c_*` single contact truncations, `s_*` shell controls, `f_*` far controls,
  `m*` multi-residue truncations), the seed tag, the ligand InChIKey and the score in kcal/mol. Empty scores are failed dockings.
- plan/: plan.json (design, roles, manifest hashes), shards*.json, ligands/*.csv (ligand sets), receptors/<TARGET>/manifest.json
  (residues removed in each variant), wt_raw/ (published DOCKSTRING receptors and boxes, Apache-2.0).
- poses/<TARGET>.tgz: seed-1 wild-type poses (PDBQT) of every ligand.
- checkpoints/: trained surrogates (PyTorch) of every condition and seed (runs_final), the scaling and kinase runs, the
  frozen baseline surrogates (baseline_*) and the hit-head networks (rsgh_wt_s*, rsgh_int_s*). Names: <condition>_s<seed>.pt,
  see code/train_dm.py for the conditions.
- runs/: per-run training curves (*_loss.csv) and validation diagnostics (*_diag.json).
- eval/: held-out predictions (summary_all.parquet), summaries (summary_all.json, paired_all.json, levels_all.json,
  scale_summary.json, kin_summary.json, hotspot.json), wild-type screening on the development and test split (wt_*),
  descriptor baselines, hit-probability screening of the baselines and of the hit-head networks (wt_hit_summary.json,
  wt_hitnet_summary.json, hit/), the budget summary of the screening tables (wt_budget_summary.json), the ML-guided docking baseline
  (ml_guided_dev.json, ml_guided_dev.log), the per-call records of the docking rate benchmark (docking_timing*.csv, docking_scores*.csv,
  dockbench*.log), throughput and independence measurements of the baseline, geometry and hit-head networks on one RTX 4090
  (throughput/thr_*; throughput_4090_run*.json is the earlier timing of the geometry networks, 87 s for the library, which thr_rsg_run*.json supersedes), the virtual scan (scan/),
  the independent Vina check (vina_check/), analysis tables (analysis/),
  the frozen reversal quadruplets (frozen/) and the log of reads of the sealed test labels (label_access_test.json).
- eval/cv/ and eval/known/: the networks and outputs of addenda 9 to 11. eval/cv holds the five-fold cross-validation over target families (runs/v4 of the repository: registries, curves, held-out
  predictions and metrics; checkpoints as checkpoints/cv_ckpt_v4_*.pt; folds in data/v4_cv_folds.json; summary in eval/v4_cv_summary.json and .md). eval/known holds the known-pocket evaluations
  (E3 on the existing checkpoints and E4 on the cross-validation checkpoints, per-pocket metrics; summaries eval/v5_known_pockets_summary.json and eval/v5_known_cv_summary.json). The full-scale
  timing (E2) is eval/v4_fair_timing.json (RTX 4090) and eval/v4_fair_timing_L40S.json with their .md tables (dockmut/v4_fair_timing.py).- eval/joint/: the joint networks of addendum 8 (runs/v3 of the repository): per-run registry files (registry_*.json), training logs, development curves
  (eval_curve_*.json), development and test metrics per seed (metrics_*, test_metrics_*), test predictions (test_predictions_*.parquet) and the
  exposure ledger of the test read. The checkpoints are in checkpoints/ as joint_ckpt_v3_<network>_s<seed>.pt. The per-stage times of the cost model
  are eval/joint_timing*.json (one laptop RTX 4060, one process per network) and eval/cost_model.json.
- data/: split definition of the preceding analysis (targets.parquet, ligand_splits.parquet, split_manifest.json) and
  the development configuration (g2_config.json).
- code/: Python code (dockmut/ scripts, src/pocketgate library).
- manuscript/: LaTeX sources, figure and table scripts of the manuscript and its Supporting Information.

## Reproduce
Script paths refer to the repository layout (`dockmut/<script>.py`); in this archive the same files sit in `code/`.
1. Regenerate receptors: `python dockmut/make_mutants.py` with the seeds of plan.json; run_shard.py verifies the manifest hashes.
2. Dock on a GPU machine with Uni-Dock 1.2.0: `dockmut/run_shard.py` (environment in dockmut/pod_setup.sh).
3. Collect and analyse: `dockmut/collect.py`, `dockmut/analyze_dockmut.py`, `dockmut/analysis_geometry.py`, `dockmut/analysis_contacts.py`.
4. Train: `python dockmut/train_dm.py --cond rsg_int --lam 3 --lr 5e-4 --n-mut 6 --mut-batch 96 --updates 20000 --seed 11 --heldout-eval`
   (conditions in `dockmut/make_jobs.py final`). Hit head: `--cond rsgh_int --lam 3 --mu 1` (or `--cond rsgh_wt --mu 1`), same other settings.
5. Evaluate: `dockmut/combine_heldout.py`, `dockmut/summarize_eval.py`, `dockmut/paired_bootstrap.py`, `dockmut/analysis_levels.py`,
   `dockmut/dm_wt_eval.py` (`--head score` or `--head hit`), `dockmut/wt_summary.py`, `dockmut/wt_hit_summary.py`,
   `dockmut/wt_hitnet_summary.py`, `dockmut/wt_budget_summary.py`, `dockmut/ml_guided_baseline.py`, `dockmut/ml_zero_shot_dual.py` (adds the dual encoders to the same json), `dockmut/dm_scan.py`,
   `scripts/v3_train_joint.py` (cross-attention and pooled networks, `--smoke` for a 300-update check), `dockmut/v3_eval.py` (`dev` and `test`),
   `dockmut/v3_reversal.py` (reversal accuracy of the joint networks from the saved test predictions and the frozen quadruplets, `eval/joint_reversal.json`),
   `dockmut/dm_joint_timing.py` (one process per network, `--only`), `dockmut/make_cost_model.py`,
   `dockmut/dm_throughput.py` (`--head hit` for the hit-head networks), `dockmut/dock_rate_bench.py` (docking rate; ligand preparation with
   `dockmut/prep_ligands.py`).
6. Addenda 9 to 11: `dockmut/v4_make_folds.py` (folds and the pod bundle), `scripts/v4_cv_train.py --jobs fold:cond:recipe:seed,...` (networks of the cross-fitted folds; it imports `v1b_train.py` for the constant-pocket conditions), `dockmut/v4_cv_analyze.py`, `dockmut/v5_known_pockets_eval.py` (E3) and `dockmut/v5_known_cv_eval.py` (E4), `dockmut/v4_fair_timing.py` (full-scale timing; `v4_fair_timing_check.py` checks the scorers).7. Build the manuscript: `manuscript/make_numbers.py`, `make_numbers_models.py`, `make_numbers_v5.py`, `make_tables.py`, `make_tables_v5.py` (with `make_tables_known.py`), the fig*.py scripts (`fig_cost.py` for Figure 2), then LaTeX.

## Environment and layout
The scripts take the parent of the folder that holds them as the project root and expect the repository layout `dockmut/`,
`src/pocketgate/`, `data/` and `configs/`. In this archive `code/` holds the files of `dockmut/` (and `code/src/pocketgate` the
library). To run a script, copy the files of `code/` into `dockmut/`, `code/src/pocketgate` into `src/pocketgate`, keep `data/` and
copy `data/g2_config.json` to `configs/g2_config.json`. Public inputs that the archive does not contain: the DOCKSTRING dataset
(scores, ligands and receptors; figshare, Apache-2.0), from which `data/processed/ligands.parquet`, the ligand-graph cache and the
label vault (`data/label_vault`, the DOCKSTRING scores split by role) are rebuilt with the scripts of `code/src/pocketgate`.
Docking environment: `code/pod_setup.sh` (micromamba, Python 3.11, Uni-Dock 1.2.0, Open Babel, RDKit). Training pods: image
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 with pip installs of rdkit, pandas, pyarrow, scipy and scikit-learn
(versions not pinned). Evaluation and analysis workstation: Python 3.13.5, PyTorch 2.7.1+cu118, scikit-learn 1.7.1, numpy 2.2.6,
pandas 2.3.1, scipy 1.16.1, CatBoost 1.2.10, RDKit 2026.03.6.
Timing pod: one RTX 4090, driver 570.169, Python 3.11.10, PyTorch 2.4.1+cu124, NumPy 2.4.6, Uni-Dock 1.2.0, Open Babel 3.2.1,
RDKit 2026.03.6. The DockMut-1 campaign was not timed call by call; the docking rate of the manuscript comes from the benchmark above.
Not in the archive: the sealed DOCKSTRING test labels, which are read only through the logged label interface (`eval/label_access_full.log`; the runs on rented GPUs read only the open cells listed in `tmp/bundle_v4/bundle_manifest.json` of the pod bundle and did not touch the test cells). Derivatives of the sealed labels in the archive are `eval/frozen/test_quads_cold.parquet` (3,536 pairwise score differences of the nine test targets from the logged reads of the preceding analysis) and the test predictions and metrics.
Figure 1 and the table-of-contents graphic are drawings that no script generates.

## Data sources and licenses
DOCKSTRING receptors, ligands and wild-type scores: Apache License 2.0 (https://github.com/dockstring/dockstring).
Derived docking scores, poses, manifests, checkpoints and code in this archive: Apache License 2.0.
