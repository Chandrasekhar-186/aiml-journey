# Day 08 — Probability & Distributions
# Date: March 20, 2026

import numpy as np
from scipy import stats

# 1. Basic probability simulation
def coin_flip_experiment(n_flips=10000):
    flips = np.random.choice(['H', 'T'], size=n_flips)
    heads_prob = np.sum(flips == 'H') / n_flips
    print(f"P(Heads) after {n_flips} flips: {heads_prob:.4f}")
    # Should converge to 0.5 — Law of Large Numbers!

coin_flip_experiment()

# 2. Bayes Theorem — spam classifier example
# P(Spam|Word) = P(Word|Spam) * P(Spam) / P(Word)
p_spam = 0.3           # 30% emails are spam
p_word_given_spam = 0.8  # 80% spam has this word
p_word_given_ham = 0.1   # 10% legit has this word
p_word = (p_word_given_spam * p_spam +
          p_word_given_ham * (1 - p_spam))

p_spam_given_word = (p_word_given_spam * p_spam) / p_word
print(f"P(Spam|Word): {p_spam_given_word:.4f}")

# 3. Normal distribution
mu, sigma = 90, 5   # mean=90, std=5
dist = stats.norm(mu, sigma)

print(f"P(score > 95): {1 - dist.cdf(95):.4f}")
print(f"P(85 < score < 95): "
      f"{dist.cdf(95) - dist.cdf(85):.4f}")

# 4. Central Limit Theorem demo
sample_means = []
for _ in range(1000):
    sample = np.random.exponential(scale=2, size=30)
    sample_means.append(np.mean(sample))

print(f"CLT Demo — Sample means follow normal dist:")
print(f"Mean of means: {np.mean(sample_means):.4f}")
print(f"Std of means:  {np.std(sample_means):.4f}")

# 5. Distributions used in ML
print("\nKey distributions:")
print(f"Normal(0,1) sample:   {stats.norm(0,1).rvs():.4f}")
print(f"Binomial(10,0.5):     {stats.binom(10,0.5).rvs()}")
print(f"Poisson(lambda=3):    {stats.poisson(3).rvs()}")
