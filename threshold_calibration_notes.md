**Calibrating the Energetic Guard: Method and Current Limitations**  
**1. What we are trying to solve**  
garde-fou-energetique.py decides whether a pair of images represents a real  
   
 scene change (transmit) or a redundant/similar view (filter, save energy).  
   
 The decision rule combines three cheap pixel-level metrics — global  
   
 histogram difference, mean per-channel intensity difference, and the  
   
 proportion of 32×32 blocks that changed — into a majority vote ("at least 2  
   
 of 3 conditions triggered → CHANGEMENT").  
The three thresholds that define each condition were originally set by hand.  
   
 The goal of this calibration exercise was to replace guesswork with  
   
 thresholds fitted against a labeled dataset.  
**2. Ground truth used**  
A separate classification pass (Qwen3.5:2b) labeled 2500 image pairs from the  
   
 AgriSense PNG dataset into four categories: *Différents* (0),  *Similaires*  
   
 (1), *Très similaires* (2),  *Identiques* (3), each with a confidence score  
   
 and a short textual justification.  
For this calibration we collapsed the labels into a binary target:  
- **CHANGEMENT** = category 0 (2362 pairs, 94.5%)  
- **similaire** = categories 1, 2, 3 (138 pairs, 5.5%)  
This mapping was chosen deliberately for the current use case: only "truly  
   
 different" pairs should trigger a transmission.  
**3. Calibration method**  
1. **Feature extraction** (calibrate_seuils.py extract) — for every  
   
 labeled pair, the three raw metrics (hist_global_diff,  
   
 mean_channel_diff, prop_blocks_changed) were recomputed directly from  
   
 the actual PNG images, using the exact same function as the deployed  
   
 guard.  
2. **Grid search** (calibrate_seuils.py optimize) — a grid of candidate  
   
 thresholds was evaluated against the "≥2 of 3" voting rule, scoring each  
   
 combination with standard binary classification metrics (precision,  
   
 recall, specificity, F1, balanced accuracy) plus a configurable  
   
 false-positive/false-negative cost.  
3. **Degenerate-threshold filtering** — combinations where a threshold sits  
   
 at the extreme of the observed data range were excluded. Such thresholds  
   
 make a condition either always-true or always-false, silently reducing  
   
 the "vote of 3" to a vote of 2 (or 1), which inflates scores without  
   
 reflecting genuine discriminative power.  
4. **Systematic sweep across rebalancing strategies and objectives**  
   
 (experiment_seuils.py) — because the "similaire" class is rare (5.5%),  
   
 raw F1-optimization risks being dominated by the majority class. Rather  
   
 than picking one rebalancing method upfront, the sweep tested all  
   
 combinations of:  
  - **no rebalancing** (baseline, raw class counts),  
  - **undersampling** the majority class (5 target ratios × 5 random  
   
 seeds, keeping the minority class intact),  
  - **class weighting** (5 target ratios, applied to precision/F1/cost  
   
 without discarding any row — recall and specificity are mathematically  
   
 unaffected by this reweighting, by construction),  
5. crossed with two optimization objectives (f1, balanced_acc).  
6. **Stability check** — for every configuration, the thresholds found were  
   
 re-applied to the complete set of 2500 pairs (not just the  
   
 subsample/weighted view used to fit them), and the same objective was  
   
 re-run across 5 different random seeds per undersampling ratio. This is  
   
 not a full held-out train/test split (see §4.2), but it does establish  
   
 whether a given threshold triple is an artifact of one particular random  
   
 subsample or a stable outcome: in this case, **undersampling vs**  
 **  
 weighting, and the exact ratio chosen, made essentially no difference to**  
 **  
 the resulting thresholds** — the only factor that meaningfully changed  
   
 the outcome was the choice of objective (f1 vs balanced_acc).  
**Results**  
| | | | | | | |  
|-|-|-|-|-|-|-|  
| **Objectif** | **hist** | **mean** | **prop** | **recall** | **specificity** | **cost (w_fp=w_fn=1)** |   
| f1 | > 0.055 | > 6.93 | > 6.64 | 0.927 | 0.196 | 284 |   
| balanced_acc | > 0.110 | > 6.93 | > 19.9 | 0.709 | 0.478 | 759 |   
   
Both rows are stable across all tested rebalancing strategies and seeds —  
   
 only prop moves meaningfully between the two objectives (6.6 → 19.9), and  
   
 that single shift accounts for essentially the entire recall/specificity  
   
 trade-off between the two rows.  
**Seuils recommandés : ** **hist > 0.110** **, ** **mean > 6.93** **, ** **prop > 19.9**  
   
 (recall ≈ 0.709, specificity ≈ 0.478), obtained under the balanced_acc  
   
 objective.  
**4. Limitations of the current calibration**  
**4.1 Severe class imbalance**  
Only 138 of 2500 pairs (5.5%) belong to the "similaire" class. Any  
   
 specificity estimate computed on ~138 examples has wide statistical  
   
 uncertainty — e.g. a specificity of 0.478 (~66/138) has an approximate 95%  
   
 confidence interval of roughly [0.39, 0.56]. Differences of a few points  
   
 between threshold candidates are within this noise band and may not reflect  
   
 a genuine improvement.  
**4.2 Selection bias from the grid search itself**  
Choosing the best-scoring combination out of thousands of candidates,  
   
 evaluated on the same 138 minority examples used to score them, is a  
   
 textbook setup for overfitting. The stability check in §3 (multiple  
   
 resampling seeds, re-evaluation on the full dataset) shows the chosen  
   
 thresholds are not an artifact of one particular random draw, but it does  
   
 **not** address this deeper issue: the full 138-example minority set was  
   
 used both to search for thresholds and to report their performance, with no  
   
 truly held-out data. Reported recall/specificity should therefore still be  
   
 treated as optimistic upper bounds rather than expected performance on  
   
 unseen pairs.  
**4.3 Possible mismatch between labeling task and guard's actual task**  
The 2500 pairs are combinatorial pairs drawn from the dataset (via  
   
 itertools.combinations), not necessarily consecutive captures from a fixed  
   
 camera position over time — which is the guard's actual operating  
   
 condition. Several "Similaires" explanations in the ground truth describe  
   
 *semantic* similarity (same type of plant, different framing/angle/context)  
   
 rather than *near-identical* framing of the same scene at two points in  
   
 time. Pixel-level metrics (histogram, channel mean, block change) are not  
   
 designed to capture semantic similarity, and cannot be expected to separate  
   
 this kind of pair well regardless of threshold choice — this is a mismatch  
   
 in what is being measured, not a tuning problem.  
**4.4 Extremely small counts in the finer-grained categories**  
Category 3 ("Identiques") has exactly 1 example, and category 2 ("Très  
   
 similaires") has 25. These are the categories closest to genuine temporal  
   
 redundancy, but there are too few of them to calibrate against directly  
   
 with any statistical confidence.  
**4.5 Only two objectives were compared, not a continuous cost sweep**  
The results table in §3 compares two discrete points (f1 and  
   
 balanced_acc), not a continuous sweep over the false-positive/false-  
   
 negative cost ratio. On this dataset, the two rows cross over in total cost  
   
 (equal weight on FP and FN) around w_fp/w_fn ≈ 13: below that ratio, the  
   
 f1 row is cheaper in aggregate; above it, the balanced_acc row is  
   
 cheaper. Where the true ratio sits relative to 13 has not been pinned down  
   
 with an actual energy/transmission cost estimate — the two rows in the  
   
 table are the only two points on that curve that have been computed so far,  
   
 not necessarily the optimum.  
**5. Suggested next steps**  
- **Sweep intermediate cost ratios** (calibrate_seuils.py optimize --objectif cost --w-fp <ratio> --w-fn 1 for several values of <ratio>)  
   
 to trace the full recall/specificity trade-off curve instead of relying  
   
 on just the two objectives compared here.  
- Build (or extract) a ground-truth set of genuinely **sequential** pairs  
   
 from the same camera position, at different capture times, to match the  
   
 guard's real operating conditions.  
- Evaluate thresholds with a proper train/test split (or k-fold  
   
 cross-validation over disjoint folds) to address the selection bias in  
   
 §4.2 — the stability check performed so far confirms robustness to  
   
 resampling method, not generalization to unseen pairs.  
- Consider restricting the "similaire" ground truth to categories 2 and 3  
   
 only (visually near-identical pairs) as a closer proxy for what the guard  
   
 is meant to detect, even though this currently leaves very few positive  
   
 examples for that class.  
