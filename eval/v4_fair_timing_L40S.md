# v4 fair timing: NVIDIA L40S (2026-10-09T08:56:20+00:00)

Library 260155 ligands, 7184423 atoms; featurization 22.37 s with 32 processes (128 CPUs). torch 2.4.1+cu124, CUDA 12.4.
P = 1000 cycles the 57 wild-type pockets (repeated pockets, timing only). All times in seconds, GPU stages best of 2, top 10% of the library selected per pocket with torch.topk for every network.

## Correctness (2000 ligands x 5 pockets, against net(ctx, lb)['logit'])

| implementation | max abs diff | Spearman (min over pockets) | top-10% overlap (min) | pass |
|---|---|---|---|---|
| dual | 3.8147e-06 | 1 | 1.0000 | True |
| pooled | 5.72205e-06 | 1.000000 | 1.0000 | True |
| xattn_padded_fp32 | 5.72205e-06 | 1 | 1.0000 | True |
| xattn_fp32 | 5.72205e-06 | 1.000000 | 1.0000 | True |
| xattn_fp16_pure | 0.0193253 | 1 | 0.995 | True |
| xattn_fp16_autocast | 0.0231628 | 1 | 0.995 | True |

## P = 1

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 22.366 | 2.396 | 1.215 | 0.0011 | 0.000273 | 0.000214 | 1.217 | 25.979 | 26.844 | 534,283,236 | 213,803 | 1.2 |
| pooled | 22.366 | 2.396 | 1.211 | 0.00113 | 0.000924 | 0.000157 | 1.213 | 25.975 | 22.374 | 240,581,814 | 214,394 | 1.2 |
| xattn_fp32 | 22.366 | 2.396 | 1.206 | 0.00122 | 0.542 | 0.000752 | 1.750 | 26.512 | 24.341 | 479,350 | 148,679 | 4.6 |
| xattn_fp16 | 22.366 | 2.396 | 1.220 | 0.00126 | 0.18 | 0.000494 | 1.401 | 26.164 | 23.776 | 1,440,663 | 185,636 | 2.8 |

## P = 5

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 22.366 | 2.396 | 1.216 | 0.00127 | 0.000273 | 0.000202 | 1.218 | 25.981 | 26.583 | 2,740,528,252 | 1,067,754 | 1.2 |
| pooled | 22.366 | 2.396 | 1.213 | 0.00122 | 0.00467 | 0.000203 | 1.219 | 25.982 | 22.961 | 266,963,006 | 1,067,266 | 1.2 |
| xattn_padded_fp32 (subset) | - | 0.164 | 0.0796 | 0.00534 | 1.131 | - | 1.216 | - | - | 88,429 | 82,251 | 1.4 |
| xattn_fp32 | 22.366 | 2.396 | 1.206 | 0.00155 | 2.779 | 0.00065 | 3.987 | 28.750 | 26.736 | 467,880 | 326,222 | 4.6 |
| xattn_fp16 | 22.366 | 2.396 | 1.217 | 0.00175 | 0.892 | 0.000594 | 2.111 | 26.874 | 24.099 | 1,456,741 | 616,095 | 2.8 |

## P = 20

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 22.366 | 2.396 | 1.214 | 0.00305 | 0.000308 | 0.000254 | 1.218 | 25.983 | 24.628 | 9,254,135,010 | 4,272,309 | 1.2 |
| pooled | 22.366 | 2.396 | 1.212 | 0.00302 | 0.0162 | 0.000293 | 1.231 | 25.999 | 22.305 | 315,587,102 | 4,225,949 | 1.3 |
| xattn_fp32 | 22.366 | 2.396 | 1.205 | 0.00411 | 11.151 | 0.000671 | 12.361 | 37.127 | 35.440 | 466,566 | 420,924 | 4.6 |
| xattn_fp16 | 22.366 | 2.396 | 1.216 | 0.00478 | 3.570 | 0.00101 | 4.792 | 29.558 | 27.661 | 1,457,119 | 1,085,788 | 2.8 |

## P = 57

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 22.366 | 2.396 | 1.220 | 0.00902 | 0.000365 | 0.000509 | 1.230 | 26.000 | 22.684 | 16,964,377,211 | 12,052,563 | 1.2 |
| pooled | 22.366 | 2.396 | 1.212 | 0.00874 | 0.0435 | 0.000645 | 1.265 | 26.036 | 22.614 | 335,530,822 | 11,725,716 | 1.3 |
| xattn_fp32 | 22.366 | 2.396 | 1.210 | 0.0118 | 32.126 | 0.000773 | 33.348 | 58.123 | 55.414 | 461,578 | 444,668 | 4.6 |
| xattn_fp16 | 22.366 | 2.396 | 1.214 | 0.0135 | 10.118 | 0.000755 | 11.347 | 36.118 | 33.358 | 1,465,474 | 1,306,880 | 2.8 |

## P = 1000

| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual | 22.366 | 2.396 | 1.215 | 0.152 | 0.00403 | 0.013 | 1.384 | 26.191 | 23.132 | 15,280,611,076 | 187,921,792 | 1.2 |
| pooled | 22.366 | 2.396 | 1.213 | 0.152 | 0.766 | 0.014 | 2.145 | 26.951 | 23.425 | 333,378,469 | 121,278,326 | 1.4 |
| xattn_fp32 | 22.366 | 2.396 | 1.211 | 0.21 | 569.829 | 0.0152 | 571.265 | 596.108 | 591.653 | 456,537 | 455,402 | 5.2 |
| xattn_fp16 | 22.366 | 2.396 | 1.215 | 0.243 | 178.173 | 0.0141 | 179.646 | 204.482 | 203.183 | 1,460,009 | 1,448,157 | 2.8 |

## Ratio to the dual encoder (model time: embedding + pocket states + scoring + top-k; end to end adds the CPU featurization and collation of the dual block)

| P | network | model time | scoring only | end to end | wall clock |
|---|---|---|---|---|---|
| 1 | pooled / dual | 1 | 2.2 | 1 | 0.8 |
| 1 | xattn_fp32 / dual | 1.4 | 1,115 | 1.0 | 0.9 |
| 1 | xattn_fp16 / dual | 1.2 | 370.9 | 1.0 | 0.9 |
| 5 | pooled / dual | 1.0 | 10.3 | 1.0 | 0.9 |
| 5 | xattn_fp32 / dual | 3.3 | 5,857 | 1.1 | 1.0 |
| 5 | xattn_fp16 / dual | 1.7 | 1,881 | 1.0 | 0.9 |
| 20 | pooled / dual | 1.0 | 29.3 | 1.0 | 0.9 |
| 20 | xattn_fp32 / dual | 10.1 | 19,835 | 1.4 | 1.4 |
| 20 | xattn_fp16 / dual | 3.9 | 6,351 | 1.1 | 1.1 |
| 57 | pooled / dual | 1.0 | 50.6 | 1.0 | 1 |
| 57 | xattn_fp32 / dual | 27.1 | 36,753 | 2.2 | 2.4 |
| 57 | xattn_fp16 / dual | 9.2 | 11,576 | 1.4 | 1.5 |
| 1000 | pooled / dual | 1.5 | 45.8 | 1.0 | 1.0 |
| 1000 | xattn_fp32 / dual | 412.7 | 33,471 | 22.8 | 25.6 |
| 1000 | xattn_fp16 / dual | 129.8 | 10,466 | 7.8 | 8.8 |

## Original padded implementation (subset) against the ragged implementations

- P = 5: ragged fp32 scoring is 5.29x and ragged fp16 16.48x the pairs/s of the padded fp32 path.
