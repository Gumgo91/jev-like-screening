# v4 fair timing: NVIDIA GeForce RTX 4090 (2026-10-09T13:18:49+00:00)

Library 260155 ligands, 7184423 atoms; featurization 39.32 s with 32 processes (96 CPUs). torch 2.4.1+cu124, CUDA 12.4.
P = 1000 cycles the 57 wild-type pockets (repeated pockets, timing only). All times in seconds, GPU stages best of 2, top 10% of the library selected per pocket with torch.topk for every network.

## Correctness (2000 ligands x 5 pockets, against net(ctx, lb)['logit'])

| implementation | max abs diff | Spearman (min over pockets) | top-10% overlap (min) | pass |
|---|---|---|---|---|
| dual | 5.72205e-06 | 1 | 1.0000 | True |
| pooled | 5.72205e-06 | 1.000000 | 1.0000 | True |
| xattn_padded_fp32 | 7.62939e-06 | 1 | 1.0000 | True |
| xattn_fp32 | 7.62939e-06 | 1.000000 | 1.0000 | True |
| xattn_fp16_pure | 0.0193253 | 1 | 0.995 | True |
| xattn_fp16_autocast | 0.0231705 | 1 | 0.995 | True |

## P = 1

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 39.322 | 2.317 | 1.592 | 0.00137 | 0.000246 | 0.000246 | 1.594 | 43.233 | 41.873 | 528,991,850 | 163,177 | 1.2 |
| pooled | 39.322 | 2.317 | 1.592 | 0.00134 | 0.000706 | 0.000191 | 1.594 | 43.233 | 36.849 | 290,275,506 | 163,161 | 1.2 |
| xattn_fp32 | 39.322 | 2.317 | 1.595 | 0.00154 | 0.468 | 0.0003 | 2.064 | 43.703 | 37.421 | 555,914 | 126,028 | 4.6 |
| xattn_fp16 | 39.322 | 2.317 | 1.623 | 0.00166 | 0.154 | 0.000336 | 1.779 | 43.418 | 38.124 | 1,683,239 | 146,221 | 2.8 |

## P = 5

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 39.322 | 2.317 | 1.597 | 0.00286 | 0.000286 | 0.00028 | 1.601 | 43.240 | 36.611 | 2,297,637,535 | 812,575 | 1.2 |
| pooled | 39.322 | 2.317 | 1.602 | 0.00175 | 0.0035 | 0.000223 | 1.607 | 43.246 | 37.621 | 349,844,126 | 809,371 | 1.2 |
| xattn_padded_fp32 (subset) | - | 0.132 | 0.117 | 0.00704 | 0.874 | - | 0.999 | - | - | 114,398 | 100,135 | 1.4 |
| xattn_fp32 | 39.322 | 2.317 | 1.590 | 0.00234 | 2.420 | 0.000538 | 4.013 | 45.652 | 38.816 | 537,435 | 324,148 | 4.6 |
| xattn_fp16 | 39.322 | 2.317 | 1.827 | 0.00263 | 0.768 | 0.000373 | 2.598 | 44.237 | 38.489 | 1,692,994 | 500,734 | 2.8 |

## P = 20

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 39.322 | 2.317 | 1.599 | 0.0044 | 0.000232 | 0.000241 | 1.604 | 43.244 | 37.830 | 10,989,660,205 | 3,244,630 | 1.2 |
| pooled | 39.322 | 2.317 | 1.599 | 0.00436 | 0.0136 | 0.000259 | 1.618 | 43.258 | 36.722 | 374,705,051 | 3,216,728 | 1.3 |
| xattn_fp32 | 39.322 | 2.317 | 1.596 | 0.00658 | 9.722 | 0.000328 | 11.324 | 52.965 | 46.517 | 535,190 | 459,459 | 4.6 |
| xattn_fp16 | 39.322 | 2.317 | 1.597 | 0.0073 | 3.072 | 0.000344 | 4.676 | 46.317 | 38.800 | 1,693,409 | 1,112,615 | 2.8 |

## P = 57

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 39.322 | 2.317 | 1.599 | 0.0121 | 0.000268 | 0.0005 | 1.612 | 43.255 | 36.699 | 19,314,380,051 | 9,198,995 | 1.2 |
| pooled | 39.322 | 2.317 | 1.599 | 0.0125 | 0.0321 | 0.000567 | 1.644 | 43.287 | 37.302 | 453,598,294 | 9,017,733 | 1.3 |
| xattn_fp32 | 39.322 | 2.317 | 1.606 | 0.0173 | 27.804 | 0.000586 | 29.427 | 71.071 | 64.064 | 533,323 | 503,911 | 4.6 |
| xattn_fp16 | 39.322 | 2.317 | 1.601 | 0.0204 | 8.749 | 0.000563 | 10.372 | 52.015 | 44.732 | 1,694,718 | 1,429,752 | 2.8 |

## P = 1000

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 39.322 | 2.317 | 1.588 | 0.207 | 0.00282 | 0.011 | 1.809 | 43.515 | 38.148 | 18,801,873,584 | 143,779,787 | 1.2 |
| pooled | 39.322 | 2.317 | 1.594 | 0.207 | 0.563 | 0.0111 | 2.376 | 44.082 | 37.810 | 452,833,797 | 109,503,748 | 1.4 |
| xattn_fp32 | 39.322 | 2.317 | 1.602 | 0.293 | 491.862 | 0.0117 | 493.769 | 535.476 | 527.616 | 528,906 | 526,876 | 5.2 |
| xattn_fp16 | 39.322 | 2.317 | 1.596 | 0.346 | 153.693 | 0.0114 | 155.647 | 197.352 | 192.334 | 1,692,568 | 1,671,444 | 2.8 |

## Ratio to the dual encoder (model time: embedding + pocket states + scoring + top-k; end to end adds the CPU featurization and collation of the dual block)

| P | network | model time | scoring only | end to end | wall clock |
|---|---|---|---|---|---|
| 1 | pooled / dual | 1.0 | 1.8 | 1.0 | 0.9 |
| 1 | xattn_fp32 / dual | 1.3 | 951.6 | 1.0 | 0.9 |
| 1 | xattn_fp16 / dual | 1.1 | 314.3 | 1.0 | 0.9 |
| 5 | pooled / dual | 1.0 | 6.6 | 1.0 | 1.0 |
| 5 | xattn_fp32 / dual | 2.5 | 4,275 | 1.1 | 1.1 |
| 5 | xattn_fp16 / dual | 1.6 | 1,357 | 1.0 | 1.1 |
| 20 | pooled / dual | 1.0 | 29.3 | 1.0 | 1 |
| 20 | xattn_fp32 / dual | 7.1 | 20,534 | 1.2 | 1.2 |
| 20 | xattn_fp16 / dual | 2.9 | 6,490 | 1.1 | 1.0 |
| 57 | pooled / dual | 1.0 | 42.6 | 1.0 | 1.0 |
| 57 | xattn_fp32 / dual | 18.3 | 36,215 | 1.6 | 1.7 |
| 57 | xattn_fp16 / dual | 6.4 | 11,397 | 1.2 | 1.2 |
| 1000 | pooled / dual | 1.3 | 41.5 | 1.0 | 1 |
| 1000 | xattn_fp32 / dual | 272.9 | 35,549 | 12.3 | 13.8 |
| 1000 | xattn_fp16 / dual | 86.0 | 11,108 | 4.5 | 5.0 |

## Original padded implementation (subset) against the ragged implementations

- P = 5: ragged fp32 scoring is 4.70x and ragged fp16 14.81x the pairs/s of the padded fp32 path.
