## Lakehouse Bronze/Silver/Gold

Bronze: raw, as-is, append-only
        → never modify, audit trail
Silver: cleaned, validated, enriched
        → quality checks applied
        → partitioned for performance
Gold:   aggregated business metrics
        → ML features, dashboards
        → rebuilt from Silver always

## Semi Join vs Anti Join
Semi:  LEFT rows WHERE match EXISTS
       → deduplication without explosion
Anti:  LEFT rows WHERE NO match
       → find orphaned records
Both:  don't add right-side columns!

## Key SQL Functions for Cert
explode():       array row → multiple rows
collect_list():  multiple rows → array
collect_set():   multiple rows → distinct array
array_contains(): check if value in array
coalesce():       first non-null value
regexp_extract(): extract with regex

## Interval Greedy Pattern
Sort by END time
Track current "coverage" end
When new start > coverage end:
    → need new coverage (arrow/room/etc)
    → update coverage to new end
Count = number of new coverages needed

## Drift Detection Pattern
Rolling window stats: mean + stddev
Z-score = (value - rolling_mean) / rolling_std
Anomaly if |z-score| > 2.5
→ Statistically significant deviation
→ Trigger retraining alert!
