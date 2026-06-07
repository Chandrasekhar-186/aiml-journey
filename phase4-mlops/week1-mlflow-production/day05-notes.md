## Week 1 MLOps Project — 7 Stages
1. Bronze→Silver→Gold data layer
2. Feature store (Delta backed)
3. Model training (MLflow tracked)
4. Quality gate (CI/CD gate)
5. A/B testing setup
6. Drift monitoring (KS test)
7. MLflow pipeline summary

## Production Checklist
✅ Signature on all registered models
✅ Feature store prevents train/serve skew
✅ Quality gate blocks bad models
✅ A/B test before full rollout
✅ Drift monitoring catches decay
✅ MLflow tracks everything

## Bit Manipulation Cheatsheet
XOR (^):   a^a=0, a^0=a, commutative
           → find unique element, swap
AND (&):   extract bits, check parity
OR (|):    set bits
NOT (~):   flip all bits
Left (<<): multiply by 2
Right (>>):divide by 2

## Sum without + operator
XOR = addition without carry
AND<<1 = carry
Repeat until no carry!

## Week 1 Complete → Week 2 Preview
Week 2: Docker + FastAPI + Databricks
        Model Serving endpoints
        Production REST API deployment
        Online inference patterns
        Real-time feature computation
