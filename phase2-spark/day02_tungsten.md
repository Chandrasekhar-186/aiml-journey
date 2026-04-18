# Tungsten Execution Engine
Date: April 12, 2026

## What is Tungsten?
Project Tungsten is Spark's execution engine
that makes Spark close to hardware performance
by bypassing JVM overhead.

## Three Key Innovations

### 1. Explicit Memory Management
Problem: JVM GC pauses kill Spark performance
Solution: Tungsten manages its OWN memory
  → Uses sun.misc.Unsafe for direct memory access
  → Stores data in binary format (not Java objects)
  → Eliminates GC pressure entirely
  → Can spill to disk efficiently

### 2. Cache-aware Computation
Problem: Random memory access = cache misses
Solution: Store data in CPU cache-friendly format
  → Sort algorithms designed for L1/L2 cache
  → Data layout matches CPU cache lines (64 bytes)
  → SIMD (vectorized) operations where possible

### 3. Whole-Stage Code Generation
Problem: Volcano iterator model = too many
         virtual function calls
Solution: Generate ONE optimized function
          for entire pipeline stage

Without Tungsten (Volcano model):
filter() → calls next() → calls next() → ...
Each next() = virtual function call = slow!

With Tungsten (whole-stage codegen):
ONE generated function handles the entire
filter → project → aggregate pipeline
No virtual calls. JIT-friendly. Fast!

## Key Metric
Tungsten made Spark 2x-10x faster
compared to Spark 1.x on same hardware!

## When Codegen Doesn't Apply
- Python UDFs (breaks codegen pipeline!)
- Complex custom aggregations
→ Always prefer built-in F.functions!
