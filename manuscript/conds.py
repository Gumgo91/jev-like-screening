"""Condition labels, groups and colours shared by figures, tables and number macros."""
from __future__ import annotations

from figstyle import PAL

# (key, short label, group)
ORDER = [
    ("C0", "Baseline", "reference"), ("C1", "Baseline, within-target rank", "reference"), ("C2", "Baseline, cross-target rank", "reference"), ("C0L", "Baseline, hit head", "reference"),
    ("size_only", "Size model", "descriptor"), ("ridge_descriptors", "Ridge", "descriptor"), ("gbm_descriptors", "Boosting", "descriptor"),
    ("ridge_descriptors_geo", "Ridge + geometry", "descriptor"), ("gbm_descriptors_geo", "Boosting + geometry", "descriptor"),
    ("gbm_pose_geo", "Boosting + geometry + pose", "pose"),
    ("b5_wt", "Dual encoder, WT", "wt"), ("rs_wt", "ResidueSum, WT", "wt"), ("rse_wt", "ResidueSum-Edit, WT", "wt"), ("rsg_wt", "ResidueSum-Geo, WT", "wt"),
    ("b5_int", "Dual encoder, interv.", "int"), ("rs_int", "ResidueSum, interv.", "int"), ("rse_int", "ResidueSum-Edit, interv.", "int"),
    ("rsg_int", "ResidueSum-Geo, interv.", "int"),
    ("rsgh_wt", "Geo + hit head, WT", "wt"), ("rsgh_int", "Geo + hit head, interv.", "int"),
    ("rsg_int_shl", "Permuted over ligands", "control"), ("rsg_int_shm", "Permuted over edits (v1)", "control"),
    ("rsg_int_sha", "All permuted (v1)", "control"), ("rsg_int_shm2", "Deranged over edits (v2)", "control"),
    ("rsg_int_sha2", "All deranged (v2)", "control"),
]
LABEL = {k: v for k, v, _ in ORDER}
GROUP = {k: g for k, _, g in ORDER}
GCOL = {"reference": PAL["grey"], "descriptor": PAL["light"], "pose": "#b9b9b9", "wt": PAL["teal"], "int": PAL["blue"], "control": PAL["orange"]}
GNAME = {"reference": "reference surrogates", "descriptor": "descriptor models", "pose": "uses wild-type pose", "wt": "wild-type training only",
         "int": "interventional training", "control": "label controls"}
# LaTeX macro suffix (letters only)
MACRO = {"C0": "CZero", "C1": "COne", "C2": "CTwo", "C0c": "CZeroC", "C0L": "CZeroL", "C1L": "COneL", "C2L": "CTwoL", "size_only": "SizeOnly", "ridge_descriptors": "RidgeDesc",
         "gbm_descriptors": "GbmDesc", "ridge_descriptors_geo": "RidgeGeo", "gbm_descriptors_geo": "GbmGeo", "gbm_pose": "GbmPose",
         "gbm_pose_geo": "GbmPoseGeo", "b5_wt": "BfiveWt", "b5_int": "BfiveInt", "rs_wt": "RsWt", "rs_int": "RsInt", "rse_wt": "RseWt",
         "rse_int": "RseInt", "rsg_wt": "RsgWt", "rsg_int": "RsgInt", "rsgh_wt": "RsghWt", "rsgh_int": "RsghInt", "rsg_int_shl": "RsgShl", "rsg_int_shm": "RsgShm", "rsg_int_sha": "RsgSha",
         "rsg_int_shm2": "RsgShmTwo", "rsg_int_sha2": "RsgShaTwo", "rsg_const": "RsgConst"}
