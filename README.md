# Instructor Effectiveness Modeling (EdTech)

**Role:** Data Science / AI Content Specialist Intern — Assignment
**Author:** Sinchana
**Deliverable:** `instructor_effectiveness.ipynb` (Google Colab / Jupyter)
**Stack:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn (no LLMs, no external data, no AutoML)

---

## 1. Problem

An EdTech platform runs the same courses across many batches taught by different instructors. Given **batch-level** data (learner outcomes, engagement, feedback), the goal is to:

1. Define what "instructor effectiveness" means
2. Aggregate batch data to **instructor level**
3. Train a classical ML model to predict an **effectiveness tier** (Low / Medium / High)
4. Interpret results and discuss limitations and ethics

There is no single correct answer. This project prioritises **reasoning, fairness and honesty about limits** over raw accuracy.

---

## 2. What makes this submission different (6 key design choices)

| # | Idea | Problem it solves |
|---|------|-------------------|
| 1 | **Course-adjusted scoring** | Some courses are simply harder. Metrics are z-scored *within each course* so an instructor isn't punished for teaching a tough subject. |
| 2 | **Noise-corrected shrinkage + asymmetric feedback reliability** | Noise-corrected tau-squared is `0.2325`, prior strength is `0.6018`, and raw-to-shrunk tier changes are `0` of 120. Low-response negative feedback is conservatively penalized. |
| 3 | **Leakage-safe target design** | If the score is built from features and the model then predicts it from the same features, accuracy is fake. The score is built from *outcome + feedback* metrics; the model learns from engagement summaries, engagement consistency, experience and versatility. |
| 4 | **Consistency and trend diagnostics** | Plain standard deviations form the engagement consistency features; CV is excluded because z-scores centre near zero. Effectiveness-score trend slope stays descriptive because `batch_id` is not a verified timeline. |
| 5 | **Repeated grouped evaluation** | Ten repeats of five `StratifiedGroupKFold` splits, grouped by instructor, report macro-F1 mean +/- std, per-class precision/recall, confusion matrix, ordinal error, majority and stratified-random baselines. |
| 6 | **Sensitivity and stability analysis** | Bootstrap tier agreement is `87.44%`; Dirichlet weight perturbations use concentration `20`. Across 200 repetitions, shrinkage clearly reduces mean absolute error from `0.2213` raw to `0.1865` shrunk, while tier flips are essentially unchanged (`0.3132` to `0.3050`). |

---

## 3. Approach

### 3.1 EDA
- Missing values, duplicates, range checks (rates within 0–1, feedback within 1–5)
- Sanity checks such as `completion_rate + dropout_rate ≈ 1`
- Distributions, correlation heatmap, batches-per-instructor histogram
- Outliers and early observations (e.g. feedback bias when response rate is low)

### 3.2 Defining Instructor Effectiveness
Effectiveness score is built from **what learners achieved and felt**, not from how busy they were:

| Component | Metrics | Weight |
|-----------|---------|--------|
| Learning outcomes | completion_rate, (1 − dropout_rate), avg_score_improvement, avg_quiz_score | 0.60 |
| Learner satisfaction | avg_feedback_score (reliability-weighted) | 0.25 |
| Consistency | 1 − normalised std of batch scores | 0.15 |

Tiers use terciles (or 25/50/25), chosen after checking class balance. Weights are assumptions and are stress-tested in the sensitivity analysis.

### 3.3 Aggregation to Instructor Level
- **Mean** for typical level, **median** for robustness to outliers
- **Std** for spread and engagement consistency, **min / max** for floor and ceiling
- **Slope** of score over batch order (batch_id used as a time proxy; caveat noted)
- **n_batches, n_courses** for experience and versatility
- **Few batches:** shrinkage and bootstrap simulations demonstrate the risk; the real data have a minimum of 7 batches per instructor, so few-batch instability is simulated rather than observed here

### 3.4 Modeling
- Logistic Regression (baseline, interpretable), Random Forest, Gradient Boosting
- Scaling for linear models, `class_weight="balanced"` for imbalance
- Feature selection via correlation pruning and permutation importance

### 3.5 Evaluation
- Macro-F1 (primary), per-class precision / recall, confusion matrix
- Precision vs recall trade-off discussion (flagging a weak instructor wrongly vs missing one)
- Comparison to a dummy baseline and an ablation (with vs without outcome-derived features)

### 3.6 Interpretation
- Permutation importance and partial-dependence plots (no external libraries)
- Plain-language findings for non-technical stakeholders
- Product ideas: instructor coaching cards, batch-allocation support, early-warning flags for mid-course drops

---

## 4. Mandatory Questions (answered in the notebook)

1. **Key features and why** — see the importance section
2. **Misleading / confounded variables** — course difficulty, feedback response bias, batch size, learner mix, completion vs dropout redundancy
3. **Real-world failure modes** — small samples, gaming of metrics, drift over time, self-selected learners, feedback bias
4. **Additional data wanted** — learner background, batch size, course difficulty, timestamps, session-level data, TA support, pre-course skill level
5. **Use for performance evaluation?** — **No, not as a sole judge.** It should be used for coaching and support, with human review, confidence intervals and instructor transparency.

---

## 5. Repository Structure

```
.
├── README.md
├── instructor_effectiveness.ipynb
├── data/
│   └── instructor_data.csv        # place the local CSV here; it is ignored by git
└── requirements.txt               # pandas, numpy, scikit-learn, matplotlib, seaborn
```

## 6. How to Run

1. Open `instructor_effectiveness.ipynb` in Google Colab or Jupyter
2. Upload the dataset CSV and set `DATA_PATH` in the first code cell
3. Run **Runtime → Run all** (a fixed `RANDOM_STATE = 42` keeps results reproducible)

## 7. Results Summary

- Tier distribution: `Low 40 (33.3%)`, `Medium 40 (33.3%)`, `High 40 (33.3%)` across 120 instructors
- Highest observed mean: `Engagement-only Logistic` | Repeated grouped macro-F1: `0.6286 +/- 0.0995` | Stratified-random chance: `0.3719 +/- 0.0853` | Majority baseline: `0.1667 +/- 0.0000`
- All four production candidates are within one highest-model standard deviation; lift versus chance is `0.2567`; mean ordinal error is `0.3775`
- Bootstrap tier agreement: `87.44%`; weight stability: `97.84%` with Dirichlet concentration `20`; raw-to-shrunk tier changes: `0`; minimum batches: `7`
- Low-confidence flag: `16` of 120 instructors under the `shrink_weight < 0.95` review threshold
- Low-confidence batch-count range: `7` to `11` batches
- Final-score ICC: `0.6326`, inflated because the consistency term is constant per instructor
- Only distinguishable held-out driver under mean > 1 std: `forum_activity_rate_mean` (`0.1721 +/- 0.0918`)
- Not distinguishable from zero: `avg_watch_time_mean` (`0.0658 +/- 0.0673`), `assignment_submission_rate_mean` (`0.0437 +/- 0.0669`), and the remaining engagement features
- Diagnostic leaky ablation: `0.8709 +/- 0.0604`; excluded because it uses outcome-derived features.
- 200-repeat small-sample simulation: shrinkage clearly reduces raw error `0.2213` -> `0.1865`, while tier flips are essentially unchanged at `0.3132` -> `0.3050`

The production feature set is leakage-safe: it uses engagement summaries, engagement consistency, experience and versatility, while the target is built from outcomes and feedback. The model is useful as a coaching signal, not as a sole judge of instructor performance. EDA found 625 completion/dropout mismatches beyond tolerance, completion `0.30` in 77 rows, dropout `0.70` in 72 rows, correlation `-0.9535`, outcome ICC `0.3095`, 67 feedback scores at 5.0, and 149 watch-time values at 1.0. The asymmetric feedback component has mean `-0.2403`, versus `0.0123` for the symmetric comparison; the bad-feedback response correlation is positive at `0.4097`.

The CSV is intentionally not committed because it is local input data. Put it at `data/instructor_data.csv` before running the notebook.

Keep the repository private if it is published; the local CSV is excluded from git history and must be supplied separately.

## 8. Limitations and Ethics

- The score is a **proxy**, not ground truth for teaching quality.
- Outcomes depend on learners, course and context, not only the instructor.
- Tiers near a boundary are uncertain and should be treated as such.
- Any use in hiring, pay or termination decisions would be inappropriate without human oversight.
