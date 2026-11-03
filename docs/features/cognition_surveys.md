# Cognitive variability and survey scoring (Phase 6)

## Reaction-time variability — `features/cognitive_variability.py`

`session_rt_variability(trials, *, min_trials=8, rt_min_ms=150.0, rt_max_multiple_of_median=3.0)`,
`compute_session_variability(cognitive, *, tz, …)` and `weekly_cognitive_summary(sessions)`.

Trials are kept when 150 ms ≤ RT ≤ 3 × the session median of the remaining trials. A session
needs ≥ 8 kept trials.

| Feature | Definition |
|---|---|
| `rt_cov` | SD (ddof 1) ÷ mean |
| `rt_tau_ms` | ex-Gaussian τ by the method of moments: τ = s·(g/2)^(1/3), with skewness g capped at 2 |
| `rt_slope_ms_per_trial` | OLS slope of RT on kept-trial order |

With non-positive skewness the ex-Gaussian does not fit, so τ is `not_applicable` and is not
set to 0. Weekly summaries are per task and local week (Monday start): the median over sessions
with an `observed`/`observed_zero` value, plus the session count.

## Survey scoring — `features/survey_scoring.py`

`SurveyScoringSpec` (pydantic) ports the legacy `scoring_dict`: `category_list`, `questions`
(text → `{category, map, aliases}`), `maps`, `reverse_items`, `item_min`/`item_max`,
`min_items`. `score_surveys(survey, spec, *, session_gap_ms=3_600_000)` gives one row per
participant × session × category.

- Item text is matched after normalisation (lower case, no punctuation, single spaces),
  including aliases such as `phq9_1`.
- An answer's value comes from its question's map first, then from `value_numeric`. Values
  outside the item range are invalid (`n_items_invalid`).
- Reverse items score `item_min + item_max − value`.
- Score = sum of valid items when ≥ `min_items` are valid. **No pro-rating**: otherwise NaN,
  `insufficient_observability`. A sum of 0 is `observed_zero`.
- Sessions are keyed by `activity_id`, else split by 1 h of silence. Duplicate answers keep the
  last.

Built-in specs (`schema_specs/survey_specs/`, shipped as package data):

| Spec | Items | Range | Minimum valid items |
|---|---|---|---|
| `phq9` | 9 (the difficulty question is not scored) | 0–27 | 7 |
| `gad7` | 7 | 0–21 | 5 |

The minimums are a choice (see `open_questions.md` #19). The published instruments do not
define a missing-item rule.

## References

- MacDonald SWS, Nyberg L, Bäckman L (2006). Intra-individual variability in behavior: links to
  brain structure, neurotransmission and neuronal activity. *Trends Neurosci* 29:474–480.
- Olivier J, Norberg MM (2010). Positively skewed data: revisiting the Box–Cox power
  transformation. *Int J Psychol Res* 3:68–75.
- Kroenke K, Spitzer RL, Williams JBW (2001). The PHQ-9. *J Gen Intern Med* 16:606–613.
- Spitzer RL, Kroenke K, Williams JBW, Löwe B (2006). A brief measure for assessing generalized
  anxiety disorder: the GAD-7. *Arch Intern Med* 166:1092–1097.
