# DockMut pre-registration (written before any held-out evaluation of interventional models)

Status date: 2026-10-01. Re-docking data for held-out targets exist; no interventional surrogate has been
evaluated on them. Family-level validation results exist only for the configuration choice described below.

## Questions
1. Do surrogates of the original study (real-pocket dual encoders, seeds 11/22/33 of C0, C1, C2) predict the
   docking-score change caused by truncating pocket residues on 19 held-out targets?
2. Does interventional training (measured changes of truncated receptors of the 38 training targets) give a
   surrogate that predicts these changes on unseen targets, and does it retain wild-type screening accuracy?

## Data
Held-out benchmark: 10 development and 9 test targets, 160 shared development ligands, up to 24 truncated
receptors per target. Training: 38 training targets (DRD2 excluded), 256 ligands, up to 24 truncated receptors.

## Primary outcome
Pearson r between predicted and measured change over all (receptor, ligand) pairs of contact truncations
(single and multi-residue) of the 19 held-out targets, with a bootstrap over targets for the 95% interval.
Secondary: mutant-level r, within-mutant r, regression gain, detection AUC of contact versus shell and far controls,
and wild-type screening (top-1% recall at 10% budget, reversal accuracy on the frozen quads of the original study).

## Conditions (three seeds 11, 22, 33 each, 20,000 updates, final checkpoint, no selection on held-out targets)
rs_wt, rs_int, rs_int_shl (labels permuted across ligands within a receptor), rs_int_shm (labels permuted across
receptors), rs_const (one constant pocket), b5_wt, b5_int; and the same five ResidueSum conditions with geometry
descriptors (rsg_*). Descriptor baselines: size_only, ridge_descriptors, gbm_descriptors.

## Configuration choice (only validation on training targets is allowed to influence it)
Validation = training on 30 training targets and evaluating on the 8 training targets of the families
gpcr_classA, cyp450 and serine_protease_S1 (all their truncated receptors). Candidates: rs_int and rsg_int with
lambda in {3, 10}, rsg_int with lambda 30, rsg_int with learning rate 1e-3, b5_int with lambda 10.
Rule: choose the candidate with the highest validation pooled r; differences below 0.02 are ties, broken toward
ResidueSum without geometry. The chosen lambda, batch shape and learning rate are then fixed for all conditions.

## Reporting
All conditions, seeds and targets are reported whatever the outcome. The held-out benchmark is evaluated once per
final model. Wild-type test labels (9 test targets) are read once, after the models are frozen, through the
guarded label API, and the access is logged. Claims of pocket reading require, at the same time, a pooled r above
the descriptor baselines and above both shuffled-label controls, with the bootstrap interval over targets excluding
the control values.

## Addendum 1 (before any held-out evaluation of an interventional model)

Two validation candidates are added to the list above: rs_int and rsg_int with lambda 10 and family-balanced sampling
of training targets (each protein family receives equal sampling mass). Predicted changes are clipped to the range of
the training labels (+-4 kcal/mol) in all evaluations. The batch shape for all candidates is six edited receptors on a
shared set of 96 ligands per update. The selection rule is unchanged. Pilot runs used to debug the pipeline (eight
training targets, six held-out targets, 3,000 to 6,000 updates) are disclosed in the Supporting Information and do
not enter any reported number.

## Addendum 2 (before any held-out evaluation of an interventional model)

The control rs_int_shm (and rsg_int_shm) assigns each edit of a training step the measured change row of another edit
of the same target evaluated on the same ligands. It destroys edit-specific information and keeps ligand main effects.
An additional control rs_int_sha (and rsg_int_sha) permutes all measured changes of a step over (edit, ligand) pairs and
keeps only their marginal distribution. The permutation across ligands within an edit (rs_int_shl) is unchanged.

## Addendum 3 (secondary analyses, written before their execution)

Secondary analyses of the selected configuration: (i) a scaling study with 4, 8, 16 and 24 random training targets
(all edits) and with 2, 4, 8 and 12 edits per training target (all targets), two seeds each; (ii) a within-family
transfer test in which four kinases (ABL1, EGFR, SRC, JAK2) are withheld from training and evaluated with all their
edits, one seed; (iii) the seed ensemble of each condition; (iv) a virtual alanine scan of a held-out target with the
full DOCKSTRING library of 260,155 ligands, reporting wall-clock time, and the agreement of the predicted mean change
per residue with the measured mean change of the docked residues; (v) an independent check of measured changes with
AutoDock Vina 1.1.2 on 32 ligands and four edits of three training targets. All are reported regardless of outcome.

## Addendum 4 (written after the first validation round, before any held-out evaluation of an interventional model)

First validation round (fold A: class A GPCRs, cytochromes P450 and S1 serine proteases withheld, 30 training targets),
reported in the Supporting Information, gave validation pooled r between -0.12 and 0.17 for ten candidates. A
gradient-boosting baseline that receives explicit edit descriptors and pocket geometry reached r 0.39 on the held-out
targets, which shows that the edit record (removed atoms, distance, change of pocket geometry) carries transferable
information. Because a single validation fold of eight targets has a large variance, the selection is repeated with two
folds: fold A as above and fold B, in which the nine nuclear receptors are withheld (29 training targets).
New candidates: rs_wt and rsg_wt (no intervention loss, reference), rsg_int with lambda 1, and a ResidueSum-Edit
architecture (rse) that adds an edit record (change of the ten geometry descriptors, removed atoms, edited residues,
distance of the edit to the box center; zero for the wild type) to the global term. rse_int is tried with lambda 3
and 10. All candidates with lambda > 0 are trained on both folds (seed 11, 20,000 updates). Rule: choose the candidate
with the highest mean validation pooled r over the two folds; differences below 0.02 are ties and are broken toward the
simpler architecture (rs before rsg before rse), then toward the larger mean receptor-level r. The chosen configuration
is fixed for all final conditions. The descriptor baselines are evaluated on both folds for reference.

## Addendum 5 (written AFTER the held-out evaluation of all 36 registered final runs; everything below is post hoc)

Registered outcome, stated before anything else: the pooled r of rsg_int on the contact edits of the 19 held-out
targets (seed ensemble 0.294, seed mean 0.247) is below that of the gradient-boosting baseline with pocket geometry
(0.394) and below both shuffled-label controls as implemented (rsg_int_shm 0.387, rsg_int_sha 0.347 for the seed
ensembles). The registered criterion for a claim of pocket reading is therefore not met.

Implementation error found while decomposing this result: the controls rsg_int_shm and rsg_int_sha used uniform random
permutations. A uniform permutation of k = 6 edit rows leaves a given edit in place with probability 1/k (one fixed
point on average per update), and a uniform permutation of all (edit, ligand) labels draws the label of an edit from
its own row with probability 1/k. Both controls therefore carry the edit-specific signal at a dilution of 1/k, which
contradicts the definition of Addendum 2 (the measured row of another edit). Corrected controls use a derangement of the
edit rows (random cyclic shift of a random edit order by 1 to k-1 positions); rsg_int_sha2 additionally permutes the
ligands independently within each row. They are run with three seeds under the registered settings and are reported next
to the first implementation. The correction was made after the held-out results were known and it makes the control
weaker, so it favors the hypothesis. The claim criterion is unchanged, and a claim of pocket reading requires that
rsg_int exceeds the descriptor baselines, the two first-implementation controls and the two corrected controls.

Post hoc analyses added after the registered outcomes were known (exploratory, labeled as such in the manuscript):
(a) decomposition of faithfulness into target level, edit level within a target and ligand level within an edit
(analysis_levels.py); (b) a descriptor model that additionally receives contact counts from the wild-type pose;
(c) a per-target analysis and a consensus of the neural ensemble with the descriptor model. No configuration is
selected from these analyses.

## Addendum 6 (written AFTER the first complete manuscript draft; post hoc)

While reviewing the screening comparison we found that the baseline surrogates had been read out with their regression
head, whereas the preceding analysis ranked ligands with their hit-probability head (a different output of the same
dual encoder). The regression head gives a lower recall (0.72 against 0.83 on the nine test targets), so a comparison of
the regression-trained networks of this study with the regression head of the baselines overstated the screening gain of the
geometry descriptors. Corrections: (a) the hit-probability readout of the baselines is reported next to the regression
readout, taken from the result files of the preceding analysis (no new label read; its development values were
reproduced with dm_baseline_logit.py); (b) the response of the baselines to edits is repeated with the hit logit
(dm_baseline_logit.py); (c) the text states that the geometry network reaches the screening recall of the hit-trained
baselines within the resolution of nine targets and does not exceed it, and that the gain of the descriptors holds for
networks trained with the same regression recipe. The registered outcomes and conditions are unchanged.

## Addendum 7 (written AFTER addendum 6 and BEFORE any model of this extension was trained; post hoc relative to the registered plan)

Motivation: the regression-trained geometry network matches but does not exceed the screening recall of the baselines that
are trained on top-1% hits (test recall 0.80 against 0.83). To test whether a Jev-like shared-state network with pocket
geometry can reach the recall of hit-trained models while keeping its response to edits, ResidueSum-Geo receives a second
readout, a hit logit computed from the same residue interaction features (modules lig_off_h, out_h, glob_h; the parameters
of the registered score head start from identical values for the same seed).

Conditions (three seeds 11, 22, 33; 20,000 updates; final checkpoint; no selection): rsgh_wt (no intervention loss) and
rsgh_int (intervention loss with lambda = 3). Loss = Huber on standardized scores + 1.0 x binary cross-entropy of the hit logit
against the top-1% hits of each training target (1st percentile of its training scores) + lambda x Huber on measured changes.
Batch shape, optimizer and learning rate are those of the registered final runs. The weight of the hit loss is fixed to 1
before training and is not tuned.

Outcomes: (1) top-1% recall at a 10% budget with the hit logit on the 10 development targets and on the 9 test targets
(frozen frames), and the reversal accuracy on the frozen quadruplets; (2) pooled r of the score head for the contact edits of the
19 held-out targets. Development metrics are computed on the pod, the test labels never leave the workstation and are read a
SECOND time for these six models only (logged in runs/label_access.log). All outcomes are reported whatever the result. A claim
that the hit-head network reaches or exceeds the recall of the hit-trained baseline requires that the 95% interval over targets
of the paired difference to the plain-objective baseline (hit head) lies at or above zero.

Further measurements without labels on a rented RTX 4090 (the GPU class of the docking runs): the independence of the score of
a ligand from the other candidates of a batch, the time to score the 260,155 ligands against all 57 wild-type pockets with the
ligand embeddings computed once, and the time of the virtual alanine scan of DHFR. They replace the laptop timings of the first
draft.

## Addendum 8 (written AFTER the revision of the manuscript to the Jev-like framing and BEFORE any model of this comparison was trained; post hoc relative to the registered plan)

Motivation: the manuscript claims that the Jev-like organization (pocket encoded once into a state, ligand embeddings that do not
depend on the pocket, a cheap readout) gives surrogates that cost far less than joint networks for a library-against-many-pockets
screen, at no loss of screening recall. Test: two joint surrogates are trained with the recipe of the baseline dual encoders
(scripts/v3_train_joint.py, outputs in runs/v3): xattn, the PocketGate network in which ligand atoms attend to the residues of
the pocket so that the ligand representation depends on the pocket, and pooled, PooledConcatNet, which concatenates pooled ligand
and pocket vectors and reads them with a multilayer perceptron.

Conditions: seeds 11, 22 and 33 for each of the two networks. Settings are those of the baseline dual encoders of runs/v2 (AdamW,
learning rate 3e-4, weight decay 1e-4, batches of 256 ligands of one training target, binary cross-entropy on the top-1% hits plus
0.2 x Huber loss on standardized scores, at most 60,000 updates or 10 epochs, evaluation every 6,094 updates, patience of three
evaluations, checkpoint = best macro Recall_1%@10% on the cold development targets). No hyperparameter is tuned and nothing is
changed after the first training run starts.

Outcomes: (1) top-1% recall at the 1, 5, 10 and 20% budgets on the 10 development targets and, after the checkpoints are frozen
(sha256 in runs/v3/v3_registry.json), on the 9 test targets with the frozen frames of the preceding analysis; the sealed test labels
are read a THIRD time, once, only for these six models (logged in runs/label_access.log); (2) the reversal accuracy on the frozen
quadruplets; (3) the time to score the 260,155 ligands against the 57 wild-type pockets and against K pocket variants on one GPU for
the joint networks (ligand atom states cached across pockets where the network allows it) and for the Jev-like networks. All
outcomes are reported whatever the result.

Claim rule: the statement that the Jev-like dual encoders reach the screening recall of the joint networks requires that the lower
limit of the 95% interval over targets of the paired difference (dual encoder with its hit-probability head minus the joint network,
test targets, 10% budget) is at or above -0.03. If the joint networks exceed this margin, the manuscript reports the cost and the
recall of both and states the trade-off.

## Addendum 9 (written AFTER the review of the manuscript that followed addendum 8 and BEFORE any job of this extension was trained or timed; post hoc relative to the registered plan)

Motivation: the review of the manuscript found three weaknesses of the comparison of addendum 8. (a) The recall at the 10% budget of the nine
test targets cannot separate networks that read the pocket from networks that do not: a network with one constant dummy pocket recovers 0.82
against 0.83, and every network, the cross-attention network included, outputs almost the same ligand ranking for every target. (b) The nine test
targets are targets on which the shared ligand ranking works, and the cross-attention network was trained once with the recipe of the dual encoders.
(c) The cost ratio of the joint network rests on a linear extrapolation of an unoptimized implementation measured on a desktop RTX 4060.
This addendum registers two measurements on rented GPUs. Neither reads the sealed test labels.

Experiment E1 (cross-validation on 49 targets, scripts/v4_cv_train.py, dockmut/v4_make_folds.py, outputs in runs/v4): the 49 targets of the training
and development roles are split into five folds of whole families (kinase; nuclear receptor; GPCR class A, CYP450 and MAOB; aspartyl protease,
metalloprotease, serine protease S1, NOS1 and PARP1; PDE5A, PTGS2 and PTPN1), the assignment has the sha256 a9551800f8660b435c0636af04c771da34868893b49dcf2ab8e3b1803b6fba9f (data/splits/v4_cv_folds.json).
For every fold the networks are trained on the targets outside the fold (training ligands of the preceding analysis, 40,000 per target) and
evaluated on the held-out targets against the 10,000 development ligands, whose labels are open cells. Early stopping and the checkpoint rule use the
macro Recall_1%@10% over six targets drawn from the training targets of the fold against 10,000 ligands of the probability-calibration split
(a deviation from the cold-development-target rule of the preceding analysis, because the folds consume the targets). Conditions: dual encoder,
pooled-concatenation network and cross-attention network, each trained with the real pockets and with one constant synthetic pocket for every
target (six conditions), recipe R0 (AdamW, learning rate 3e-4, weight decay 1e-4, batches of 256 ligands of one target, binary cross-entropy on
the top-1% hits plus 0.2 x Huber loss on standardized scores, at most 60,000 updates or 10 epochs, evaluation every 6,094 updates, patience of
three evaluations), seed 11 first and seed 22 added if the first seed finishes within the budget of the rented time. A second recipe R1 (learning
rate 1e-4, patience of eight evaluations, at most 150,000 updates or 30 epochs) is run for the dual encoder and the cross-attention network with
the real pockets, to test whether the cross-attention network was undertrained. No hyperparameter is tuned and nothing is changed after the first
job starts.

Outcomes of E1, all reported whatever the result: (1) top-1% recall at the 1, 5, 10 and 20% budgets per held-out target and as the macro mean over
the 49 targets, with 95% percentile intervals from 5,000 bootstrap resamples of the targets; (2) the pocket gain of each architecture, the paired
difference between the network and its constant-pocket twin; (3) the selectivity, the correlation of the ligand-centered predictions with the
ligand-centered docking scores within a fold, and the reversal accuracy on quadruplets drawn within each fold; (4) the same outcomes on the targets
whose shared-ranking oracle recall (the recall of the ranking by the mean docking score over the training targets) is below 0.5.

Claim rules of E1: (a) the statement that the dual encoder reaches the screening recall of a joint network on 49 targets requires that the lower
limit of the 95% interval of the paired difference (dual encoder minus network, 10% budget) is at or above -0.03, for the cross-attention network
in recipes R0 and R1 and for the pooled network; (b) a network is said to read the pocket only if the lower limit of its pocket gain at the 10%
budget is above zero and its selectivity exceeds that of the constant-pocket twin (zero by construction); (c) otherwise the manuscript states that the
network does not read the pocket under this recipe.

Experiment E2 (cost at the full scale, dockmut/v4_fair_timing.py): on one rented RTX 4090 the complete DOCKSTRING library of 260,155 ligands is
screened from SMILES against 1, 5, 20 and 57 wild-type pockets, and against 1,000 pockets obtained by cycling the 57 pockets, with the dual encoder
(matrix product), the pooled network (decomposed first layer) and the cross-attention network in its original padded float32 form, in a ragged float32
form and in a ragged float16 form, after a check that every optimized scorer reproduces the reference module (rank correlation of at least 0.9999 in
float32 and 0.999 in float16). The stages (featurization, embedding, pocket states, scoring) and the end-to-end time are reported as measured, and the
ratio of the cross-attention network to the dual encoder replaces the extrapolated ratio of addendum 8 in the manuscript, whatever its value.

## Addendum 10 (written AFTER the experiments of addendum 9 and BEFORE the known-pocket evaluation was computed; post hoc relative to the registered plan)

Scope decision: the manuscript is restricted to the pockets of the training data (known pockets), the setting in which the surrogates are used
as multi-pocket screening engines after a docked training sample exists for each pocket. The cross-validation on held-out target families of
addendum 9 (runs/v4, dockmut/eval/v4_cv_summary.json) was completed and is kept in the data archive; it is not part of the Results of the
manuscript, which states the restriction to known pockets in the Discussion and lists addendum 9 in the Supporting Information.

Experiment E3 (known-pocket evaluation, dockmut/v5_known_pockets_eval.py, outputs in runs/v5): the existing checkpoints of the dual encoder
(runs/v1 and runs/v2, seeds 11, 22 and 33), the dual encoder with one constant synthetic pocket (three seeds), the pooled-concatenation network
and the cross-attention network (runs/v3, three seeds each) are evaluated on the 39 training targets against the 10,000 development ligands
(labels of the open cell train x dev; no sealed label is read). The development ligands were never training ligands of these networks; the
checkpoint rule of these networks used the cold development targets only. Outcomes, all reported whatever the result: top-1% recall at the 1, 5, 10
and 20% budgets per target and as the macro mean over the 39 targets (mean over seeds per target), with 95% percentile intervals from 5,000
bootstrap resamples of the targets, and paired differences between networks; the model times of addendum 9 (experiment E2) give the cost side.
The manuscript states the recall of the shared-state networks as a fraction of the recall of the cross-attention network with its interval.

## Addendum 11 (written AFTER the outcomes of experiment E3 of addendum 10 were known and BEFORE the cross-fitted known-pocket evaluation was computed; post hoc)

Reason: the checkpoint rule of the networks of experiment E3 (macro recall on the cold development targets) selects checkpoints for new targets, and it can
select early checkpoints that do not suit the pockets of the training data. E3 gave top-1% recall at the 10% budget of 0.860 (dual encoder), 0.849 (pooled
network), 0.847 (cross-attention network) and 0.833 (constant pocket) on the 39 training pockets. The networks of the cross-validation runs of addendum 9 choose their
checkpoint on seen targets (six validation targets of the training targets of the fold against 10,000 calibration ligands) and therefore suit the known-pocket setting.
Experiment E4 (dockmut/v5_known_cv_eval.py, outputs in runs/v5): every network of a fold (conditions dual, pooled and cross-attention with the real pockets and with the constant
pocket, recipe R0, and the recipe R1 for the dual encoder and the cross-attention network, seed 11) is evaluated on the pockets of its training targets against the 10,000
development ligands (open labels; neither training nor checkpoint choice used them). The recall of a pocket is the mean over the folds in which the pocket is a training target
(four of five); summaries are macro means over the 49 pockets with 95% bootstrap intervals and paired differences, as in E3. Both E3 and E4 are reported whatever the result.
