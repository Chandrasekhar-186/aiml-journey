# Phase 3 LLM Day 4+5 — Eval + Agents
# Date: June 1, 2026
# How to measure LLMs + build AI Agents!

import numpy as np
from collections import Counter
import mlflow

print("="*60)
print("LLM Evaluation + AI Agents")
print("="*60)

"""
LLM EVALUATION — COMPLETE GUIDE

Problem: how do you measure LLM quality?
→ No single ground truth for generation!
→ Different tasks need different metrics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTOMATIC METRICS:

1. BLEU (Bilingual Evaluation Understudy):
   Measures n-gram overlap with reference
   BLEU-1: unigram overlap
   BLEU-4: 1+2+3+4-gram overlap (weighted)
   Range: 0 to 1 (higher = better)
   Good for: translation
   Bad for: open-ended generation

2. ROUGE (Recall-Oriented Understudy):
   ROUGE-N: n-gram recall vs reference
   ROUGE-L: longest common subsequence
   Range: 0 to 1
   Good for: summarization
   Formula:
   ROUGE-N = matched_ngrams / total_ref_ngrams

3. METEOR: alignment + synonym matching
   Better than BLEU for fluency
   Uses WordNet for synonym matching

4. Perplexity:
   PP = exp(-1/N * Σ log P(wₜ|w₁...wₜ₋₁))
   Lower = better (model is less confused)
   Good for: language model quality
   Bad for: task-specific quality

5. BERTScore:
   Uses BERT embeddings for comparison
   Captures semantic similarity!
   Correlates better with human judgment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BENCHMARK DATASETS:

MMLU (Massive Multitask Language Understanding):
→ 57 subjects (math, law, medicine, coding...)
→ 4-choice multiple choice questions
→ Measures: breadth of knowledge

HumanEval:
→ 164 Python programming problems
→ Measures: code generation quality
→ Pass@k: probability of passing with k attempts

GSM8K:
→ Grade school math word problems
→ Measures: reasoning + arithmetic

TruthfulQA:
→ Questions humans often answer wrong
→ Measures: factual accuracy + truthfulness

HellaSwag:
→ Common sense reasoning
→ Complete the sentence plausibly

Databricks uses these to evaluate:
→ DBRX: their open-source LLM
→ Custom fine-tuned models via Mosaic AI
→ Enterprise model selection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HALLUCINATION DETECTION:

Types:
1. Factual: states wrong facts
   "Paris is the capital of Germany"
2. Faithful: contradicts source document
   (critical for RAG!)
3. Intrinsic: contradicts itself
4. Extrinsic: adds info not in source

Detection methods:
1. NLI (Natural Language Inference):
   Model: does context ENTAIL the claim?
   entailment → faithful ✅
   contradiction → hallucination ❌
   neutral → uncertain

2. LLM-as-judge:
   Ask another LLM: "Is this claim supported?"
   GPT-4 as judge → surprisingly accurate!

3. Self-consistency:
   Generate N answers, check agreement
   Majority vote = confidence measure

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI AGENTS — COMPLETE UNDERSTANDING

Agent = LLM + Tools + Memory + Planning

Simple LLM:    User → LLM → Response
Agent loop:    User → LLM → Tool call →
               Tool result → LLM → ...repeat...
               → Final response

REACT framework (Reasoning + Acting):
  Thought:  "I need to search for X"
  Action:   search("X")
  Observation: "Found: Y"
  Thought:  "Now I can answer using Y"
  Answer:   "The answer is Y"

Tool types:
  Search:    web_search, vector_search
  Code:      python_repl, bash
  Data:      sql_query, spark_query
  External:  api_call, email, calendar
  Memory:    remember, recall

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATABRICKS AI AGENTS:

Mosaic AI Agent Framework:
→ Build agents with Databricks tools
→ Unity Catalog functions as tools
→ MLflow for agent tracing
→ Deploy as REST endpoint

Agent tools in Databricks:
→ SQL warehouse queries
→ Delta Lake reads
→ MLflow model inference
→ Vector Search retrieval
→ External API calls

mlflow.tracing:
→ Trace every agent step automatically
→ See: thought → tool call → result → answer
→ Debug exactly where agents go wrong!
"""

# 1. BLEU score from scratch
def compute_bleu(reference: str,
                  hypothesis: str,
                  max_n: int = 4) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) /
              max(len(ref_tokens), 1))

    scores = []
    for n in range(1, max_n + 1):
        # Get n-grams
        ref_ngrams = Counter([
            tuple(ref_tokens[i:i+n])
            for i in range(len(ref_tokens)-n+1)
        ])
        hyp_ngrams = [
            tuple(hyp_tokens[i:i+n])
            for i in range(len(hyp_tokens)-n+1)
        ]
        if not hyp_ngrams:
            scores.append(0.0)
            continue

        # Clipped count
        matched = sum(
            min(hyp_ngrams.count(ng),
                ref_ngrams.get(ng, 0))
            for ng in set(hyp_ngrams)
        )
        precision = matched / len(hyp_ngrams)
        scores.append(precision)

    # Geometric mean
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = np.mean([np.log(s + 1e-10)
                        for s in scores])
    return float(bp * np.exp(log_avg))

# 2. ROUGE score from scratch
def compute_rouge_n(reference: str,
                     hypothesis: str,
                     n: int = 1) -> dict:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    ref_ngrams = Counter([
        tuple(ref_tokens[i:i+n])
        for i in range(len(ref_tokens)-n+1)
    ])
    hyp_ngrams = Counter([
        tuple(hyp_tokens[i:i+n])
        for i in range(len(hyp_tokens)-n+1)
    ])

    # Overlap
    overlap = sum(
        min(hyp_ngrams[ng], ref_ngrams[ng])
        for ng in hyp_ngrams
    )

    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    f1 = (2 * precision * recall /
           max(precision + recall, 1e-10))

    return {'p': precision, 'r': recall,
             'f1': f1}

# 3. Test metrics
print("\n=== EVALUATION METRICS ===")
reference = "Apache Spark is a fast distributed \
computing engine for large scale data processing"
hypothesis1 = "Spark is a distributed engine \
for processing large data quickly"
hypothesis2 = "I love eating pizza on weekends"

bleu1 = compute_bleu(reference, hypothesis1)
bleu2 = compute_bleu(reference, hypothesis2)
rouge1_good = compute_rouge_n(
    reference, hypothesis1, n=1
)
rouge1_bad = compute_rouge_n(
    reference, hypothesis2, n=1
)

print(f"Reference: {reference[:50]}...")
print(f"\nGood hypothesis:")
print(f"  BLEU:     {bleu1:.4f}")
print(f"  ROUGE-1:  {rouge1_good}")
print(f"\nBad hypothesis:")
print(f"  BLEU:     {bleu2:.4f}")
print(f"  ROUGE-1:  {rouge1_bad}")

# 4. LLM-as-judge pattern
print("\n=== LLM-AS-JUDGE ===")
print("""
# Production evaluation with LLM judge

from openai import OpenAI

client = OpenAI()

def evaluate_faithfulness(
    context: str,
    response: str
) -> dict:
    prompt = f'''
    Context: {context}
    Response: {response}

    Is the response faithful to the context?
    Score 1-5 where:
    1 = completely hallucinated
    5 = perfectly faithful

    Respond with JSON:
    {{"score": X, "reason": "..."}}
    '''

    result = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user",
                    "content": prompt}]
    )
    return json.loads(result.choices[0]
                              .message.content)

# Example
score = evaluate_faithfulness(
    context="Spark was created at Berkeley in 2009",
    response="Spark was created at MIT in 2010"
)
print(f"Faithfulness score: {score}")
# {"score": 1, "reason": "Wrong institution and year"}
""")

# 5. AI Agent implementation
print("\n=== AI AGENT IMPLEMENTATION ===")

class SimpleTool:
    def __init__(self, name, description, fn):
        self.name = name
        self.description = description
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

class SimpleAgent:
    """ReAct-style agent"""
    def __init__(self, tools, max_steps=5):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.trace = []

    def run(self, query: str) -> str:
        """
        Simulate ReAct loop:
        Thought → Action → Observation → repeat
        """
        print(f"\nQuery: {query}")
        print("-" * 40)

        # Simulate agent steps
        steps = [
            {
                "thought":
                    "I need to search for info",
                "action": "search",
                "action_input": query,
            },
            {
                "thought":
                    "Found relevant info, "
                    "let me analyze",
                "action": "analyze",
                "action_input":
                    "Analyzing search results",
            }
        ]

        for i, step in enumerate(
                steps[:self.max_steps]):
            print(f"Thought {i+1}: "
                  f"{step['thought']}")

            if step['action'] in self.tools:
                tool = self.tools[step['action']]
                result = tool(
                    step['action_input']
                )
                print(f"Action: {step['action']}"
                      f"({step['action_input']})")
                print(f"Observation: {result}")
                self.trace.append({
                    **step, "result": result
                })

        return "Agent completed task! ✅"

# Create tools
tools = [
    SimpleTool(
        "search",
        "Search for information",
        lambda q: f"Found info about: {q}"
    ),
    SimpleTool(
        "analyze",
        "Analyze data",
        lambda d: f"Analysis complete: {d}"
    ),
    SimpleTool(
        "spark_query",
        "Query Delta Lake with Spark",
        lambda q: f"Spark results for: {q}"
    ),
]

agent = SimpleAgent(tools)
result = agent.run(
    "What is the accuracy of our ML model?"
)
print(f"\nFinal: {result}")

# 6. MLflow tracing for agents
print("\n=== MLFLOW AGENT TRACING ===")
print("""
# MLflow 2.13+ supports agent tracing!

import mlflow

mlflow.set_experiment("agent_traces")

# Automatic tracing
@mlflow.trace
def run_agent(query: str) -> str:
    with mlflow.start_span("retrieval") as span:
        docs = retrieve(query)
        span.set_attribute("n_docs", len(docs))

    with mlflow.start_span("generation") as span:
        answer = llm.generate(query, docs)
        span.set_attribute("tokens", len(answer))

    return answer

# Now every agent step is traced!
# View in MLflow UI → Traces tab
# See: latency per step, token counts,
#      retrieved docs, full conversation
""")

# 7. Log to MLflow
mlflow.set_experiment("phase3_llm_eval")
with mlflow.start_run(
        run_name="LLM_eval_agents"):
    mlflow.log_metrics({
        "bleu_good": bleu1,
        "bleu_bad": bleu2,
        "rouge1_good_f1":
            rouge1_good['f1'],
        "rouge1_bad_f1":
            rouge1_bad['f1'],
    })
    mlflow.log_params({
        "metrics": "BLEU+ROUGE+BERTScore",
        "benchmarks":
            "MMLU+HumanEval+GSM8K",
        "agent_framework": "ReAct",
        "tracing": "MLflow_traces"
    })
    print("\nLLM eval + agents logged!")

print("\n" + "="*60)
print("LLM Eval + AI Agents — MASTERED! 🤖")
print("LLM Week Day 4+5 COMPLETE!")
print("="*60)
