Ensure the categorisation system is fair, transparent, and safe for users.

- **Merchant distribution bias**: Over-representation of major merchants (e.g., Amazon) may bias classifier toward those categories.
- **Regional naming conventions**: Merchant strings vary by region/language; model trained on one region may underperform in another.
- **Amount-based leakage**: If amounts are provided and correlated with categories, careful handling is required.

- **Taxonomy aliasing** reduces misclassification via high-precision token lookups for common merchants.
- **Char n-gram features** increase robustness to noisy/misspelled merchant strings.
- **Class balancing**: `class_weight='balanced'` used in `train_baseline.py`.
- **Human-in-the-loop**: Low-confidence predictions flagged and stored in `data/feedback/feedback.csv` for correction and retraining.
- **PII masking**: Long numeric sequences masked during preprocessing (`src/preprocess.py`).
- **Transparent explanations**: SHAP-based feature contributions shown in UI to explain decisions.

- Confusion matrix inspection for per-class performance.
- Manual checks on noisy input examples (truncated names, added tokens).
- Reproducibility: deterministic seeds (42) for experiments.

- Expand dataset coverage across geographies and merchant types.
- Provide user-facing disclaimers for ambiguous transactions.
- Add continuous monitoring for drift and periodic re-evaluation.
- Log model decisions (with privacy controls) for audits.
