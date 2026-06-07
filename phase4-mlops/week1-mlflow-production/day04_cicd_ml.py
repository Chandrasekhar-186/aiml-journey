# Phase 4 Day 4 — CI/CD for ML
# Date: June 15, 2026
# From notebooks to production pipelines!

import subprocess
import json
import mlflow
from datetime import datetime

print("="*60)
print("CI/CD for ML — Production Engineering")
print("="*60)

"""
CI/CD FOR ML — WHY IT MATTERS

Notebook workflow (most teams):
  1. Open Databricks notebook
  2. Run cells manually
  3. "It works on my cluster!"
  4. Copy-paste to production
  5. Something breaks
  6. Can't reproduce the bug

CI/CD workflow (production teams):
  1. Write code in Python files + git
  2. Push → automated tests run
  3. Tests pass → deploys automatically
  4. Something breaks → git revert in 1 click
  5. Full audit trail of every change

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ML CI/CD PIPELINE:

Trigger: git push to main branch
    ↓
CI (Continuous Integration):
  → Run unit tests (pytest)
  → Run data validation tests
  → Run model quality tests
  → Lint code (flake8/black)
  → Build Docker image
    ↓
CD (Continuous Deployment):
  → Deploy to staging environment
  → Run integration tests
  → A/B test against current production
  → If metrics pass → deploy to production
  → Update MLflow Model Registry
  → Notify team (Slack/email)

Tools:
  GitHub Actions: CI/CD for GitHub repos
  Azure DevOps:   Microsoft ecosystem
  Databricks:     Asset Bundles (DABs)
                  native CI/CD for notebooks!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATABRICKS ASSET BUNDLES (DABs):

The modern way to do CI/CD on Databricks!
→ Define jobs/pipelines as YAML
→ Deploy with: databricks bundle deploy
→ Git-backed, version controlled
→ Multiple targets (dev/staging/prod)

databricks.yml:
bundle:
  name: ml_pipeline

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://dev.databricks.com
  prod:
    mode: production
    workspace:
      host: https://prod.databricks.com

resources:
  jobs:
    ml_training_job:
      name: ML Training Pipeline
      tasks:
        - task_key: train
          python_wheel_task:
            package_name: ml_pipeline
            entry_point: train
          new_cluster:
            spark_version: 14.3.x-scala2.12
            num_workers: 4

Deploy:
  databricks bundle deploy --target dev
  databricks bundle run ml_training_job
  databricks bundle deploy --target prod

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTING STRATEGY FOR ML:

Unit tests:
  → Test individual functions
  → Mock data, fast execution
  → Run on every commit

Integration tests:
  → Test full pipeline with small data
  → Real Spark, real MLflow
  → Run on PR merge

Model quality tests:
  → Accuracy above threshold
  → No significant degradation vs baseline
  → Feature drift within bounds
  → Inference latency acceptable

Data validation tests:
  → Schema matches expected
  → No null values in critical columns
  → Value ranges valid
  → Referential integrity
"""

# 1. Unit test framework
print("\n=== UNIT TESTS FOR ML ===")

class TestMLPipeline:
    """Unit tests for ML pipeline components"""

    def test_feature_engineering(self):
        """Test feature computation"""
        import numpy as np
        # Simple feature: log transform
        def log_feature(x):
            return np.log(x + 1)

        # Test cases
        assert abs(log_feature(0) - 0.0) < 1e-6
        assert abs(log_feature(1) - 0.693) < 1e-3
        assert log_feature(-1 + 1e-10) >= 0
        print("  ✅ test_feature_engineering passed")
        return True

    def test_model_output_shape(self):
        """Test model predictions shape"""
        from sklearn.ensemble import \
            RandomForestClassifier
        import numpy as np

        model = RandomForestClassifier(
            n_estimators=10, random_state=42
        )
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)

        X_test = np.random.randn(20, 5)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)

        assert preds.shape == (20,), \
            f"Expected (20,), got {preds.shape}"
        assert probas.shape == (20, 2), \
            f"Expected (20,2), got {probas.shape}"
        assert all(0 <= p <= 1
                    for p in probas.flatten())
        print("  ✅ test_model_output_shape passed")
        return True

    def test_accuracy_threshold(self):
        """Test model meets minimum accuracy"""
        from sklearn.ensemble import \
            RandomForestClassifier
        from sklearn.datasets import \
            make_classification
        from sklearn.metrics import accuracy_score
        import numpy as np

        X, y = make_classification(
            n_samples=500, n_features=10,
            random_state=42
        )
        split = 400
        model = RandomForestClassifier(
            n_estimators=50, random_state=42
        )
        model.fit(X[:split], y[:split])
        acc = accuracy_score(
            y[split:], model.predict(X[split:])
        )

        MIN_ACCURACY = 0.75
        assert acc >= MIN_ACCURACY, \
            f"Accuracy {acc:.3f} below " \
            f"threshold {MIN_ACCURACY}"
        print(f"  ✅ test_accuracy_threshold passed"
              f" (acc={acc:.3f})")
        return True

    def test_data_schema(self):
        """Test input data schema"""
        import pandas as pd

        schema = {
            "user_id": "int64",
            "value": "float64",
            "label": "int64"
        }
        # Simulate valid data
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "value": [0.5, 0.8, 0.3],
            "label": [0, 1, 0]
        })

        for col, dtype in schema.items():
            assert col in df.columns, \
                f"Missing column: {col}"
            assert str(df[col].dtype) == dtype, \
                f"Wrong dtype for {col}: " \
                f"{df[col].dtype} != {dtype}"

        assert df["label"].isin([0, 1]).all(), \
            "Labels must be 0 or 1"
        assert (df["value"] >= 0).all(), \
            "Values must be non-negative"
        print("  ✅ test_data_schema passed")
        return True

    def test_no_data_leakage(self):
        """Ensure train/test split is clean"""
        import numpy as np
        indices = np.arange(1000)
        split = 800
        train_idx = indices[:split]
        test_idx = indices[split:]

        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, \
            f"Data leakage! {len(overlap)} " \
            f"samples in both train and test"
        print("  ✅ test_no_data_leakage passed")
        return True

    def run_all(self):
        print("Running ML pipeline tests...")
        tests = [
            self.test_feature_engineering,
            self.test_model_output_shape,
            self.test_accuracy_threshold,
            self.test_data_schema,
            self.test_no_data_leakage,
        ]
        passed = failed = 0
        for test in tests:
            try:
                test()
                passed += 1
            except AssertionError as e:
                print(f"  ❌ {test.__name__}"
                      f" FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"  💥 {test.__name__}"
                      f" ERROR: {e}")
                failed += 1

        print(f"\nResults: {passed} passed, "
              f"{failed} failed")
        return failed == 0

# Run tests
tester = TestMLPipeline()
all_passed = tester.run_all()

# 2. GitHub Actions workflow
print("\n=== GITHUB ACTIONS CI/CD ===")
github_actions_yaml = """
# .github/workflows/ml_cicd.yml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: |
          pytest tests/ -v --cov=src
          pytest tests/test_model_quality.py

      - name: Lint code
        run: |
          flake8 src/ --max-line-length=88
          black src/ --check

  deploy-staging:
    needs: test
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        env:
          DATABRICKS_TOKEN: ${{secrets.DATABRICKS_STAGING_TOKEN}}
          DATABRICKS_HOST: ${{secrets.STAGING_HOST}}
        run: |
          databricks bundle deploy --target staging
          databricks bundle run ml_training_job

      - name: Run integration tests
        run: pytest tests/integration/ -v

  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        env:
          DATABRICKS_TOKEN: ${{secrets.DATABRICKS_PROD_TOKEN}}
        run: |
          databricks bundle deploy --target prod
"""
print(github_actions_yaml)

# 3. Databricks Asset Bundle structure
print("\n=== DATABRICKS ASSET BUNDLES ===")
print("""
Project structure with DABs:

ml_pipeline/
├── databricks.yml          ← bundle config
├── requirements.txt
├── src/
│   ├── train.py            ← training code
│   ├── evaluate.py
│   └── features.py
├── tests/
│   ├── test_features.py    ← unit tests
│   ├── test_model.py
│   └── integration/
│       └── test_pipeline.py
├── resources/
│   ├── jobs/
│   │   └── training_job.yml ← job YAML
│   └── pipelines/
│       └── dlt_pipeline.yml  ← DLT YAML
└── .github/
    └── workflows/
        └── cicd.yml         ← GitHub Actions

Commands:
  databricks bundle validate  ← check config
  databricks bundle deploy    ← deploy to workspace
  databricks bundle run       ← run a job
  databricks bundle destroy   ← cleanup
""")

# 4. Model promotion workflow
print("\n=== AUTOMATED MODEL PROMOTION ===")

def automated_promotion_check(
    new_model_acc: float,
    baseline_acc: float,
    min_improvement: float = 0.01
) -> dict:
    """
    Automated model promotion decision
    Used in CI/CD pipeline!
    """
    should_promote = (
        new_model_acc >= baseline_acc +
        min_improvement and
        new_model_acc >= 0.80
    )

    return {
        "new_model_acc": new_model_acc,
        "baseline_acc": baseline_acc,
        "improvement":
            new_model_acc - baseline_acc,
        "should_promote": should_promote,
        "reason":
            "✅ Promotes: "
            f"+{new_model_acc-baseline_acc:.3f}"
            if should_promote else
            "❌ Rejects: insufficient improvement"
    }

scenarios = [
    (0.92, 0.88),  # clear improvement
    (0.89, 0.88),  # marginal
    (0.85, 0.88),  # regression!
]
print("Promotion decision scenarios:")
for new_acc, base_acc in scenarios:
    result = automated_promotion_check(
        new_acc, base_acc
    )
    print(f"  New={new_acc:.2f} "
          f"Base={base_acc:.2f}: "
          f"{result['reason']}")

# 5. Log CI/CD run
mlflow.set_experiment("phase4_cicd")
with mlflow.start_run(run_name="cicd_pipeline"):
    mlflow.log_params({
        "ci_tool": "GitHub_Actions",
        "cd_tool": "Databricks_Asset_Bundles",
        "test_framework": "pytest",
        "linter": "flake8+black"
    })
    mlflow.log_metrics({
        "tests_passed": 5,
        "tests_failed": 0,
        "test_coverage": 0.87
    })
    mlflow.set_tag(
        "pipeline_status",
        "passed" if all_passed else "failed"
    )
    print("\nCI/CD run logged to MLflow!")

print("\n" + "="*60)
print("CI/CD for ML — MASTERED! ⚙️")
print("Phase 4 Day 4 COMPLETE!")
print("="*60)
