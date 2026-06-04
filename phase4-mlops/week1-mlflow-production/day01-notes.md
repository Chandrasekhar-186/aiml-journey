## MLflow Production Stack

Training:  log_params, log_metrics,
           log_model, register_model
Registry:  None → Staging → Production
Serving:   mlflow models serve -m URI -p PORT
           OR Databricks Model Serving
Inference: POST /invocations with JSON

## Model Signature (CRITICAL!)
infer_signature(X_train, model.predict(X_train))
→ Validates input/output schema at serving
→ Prevents schema mismatch in production!
→ Always include for registered models!

## pyfunc Custom Model
class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        # model_input: pandas DataFrame
        return predictions_df

Benefits:
→ Package preprocessing + model together
→ No training-serving skew!
→ Works with any framework
→ Serves via same REST API

## A/B Testing Pattern
Traffic split: 80% model A, 20% model B
Track: predictions per model + accuracy
Compare: after N requests, pick winner
MLflow: log traffic + accuracy per version

## Find Peak Binary Search
nums[mid] > nums[mid+1]: peak in left
else: peak in right
Key: boundary nums[-1] = nums[n] = -inf
     → always a peak exists!

## Phase 4 Week 1 Plan
Day 1: MLflow serving + A/B testing ✅
Day 2: Feature stores + drift detection
Day 3: Databricks Workflows + DLT
Day 4: Delta Live Tables + Autoloader
Day 5: CI/CD for ML
Day 6: Week 1 project
Day 7: Week 1 review
