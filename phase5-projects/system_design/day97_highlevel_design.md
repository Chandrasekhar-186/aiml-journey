# High-Level System Design — Databricks Interview Prep
Date: July 5, 2026

## SYSTEM 1: Bookstore Service
(Confirmed asked in 2026 Databricks interviews!)

### Requirements
- Customer sends book request + credit card +
  max price
- Query 50-200 bookstores via their APIs
- Find lowest price for requested book
- If lowest < max price → charge card
- Else → return lowest price found
- Latency: 10-20 seconds acceptable
- Scale: relatively small

### High-Level Architecture
Client
↓ POST /find-book {title, cc, max_price}
API Gateway (FastAPI)
↓ validate input + auth
BookSearchOrchestrator
├── ThreadPoolExecutor (parallel API calls!)
│     ├── BookstoreAPI_1.search(title)
│     ├── BookstoreAPI_2.search(title)
│     ├── ... (up to 200 sellers)
│     └── BookstoreAPI_N.search(title)
↓ collect results (timeout=15s)
PriceAggregator
↓ find_min_price(results)
PriceDecisionEngine
├── if min_price <= max_price:
│     PaymentService.charge(cc, min_price)
│     → return {status: "purchased", price}
└── else:
→ return {status: "not_found",
 lowest_price: min_price}

 ### Low-Level Running Code
```python
import asyncio
import aiohttp
from concurrent.futures import (
    ThreadPoolExecutor, as_completed
)
import time
from typing import List, Optional

class BookstoreClient:
    def __init__(self, url: str,
                  timeout: float = 12.0):
        self.url = url
        self.timeout = timeout

    def search(self, title: str
               ) -> Optional[float]:
        """Returns price or None if not found"""
        try:
            import time, random
            # Simulate API call latency
            time.sleep(random.uniform(0.1, 8.0))
            # Simulate found/not found
            if random.random() > 0.3:
                return round(
                    random.uniform(5.0, 50.0), 2
                )
            return None
        except Exception:
            return None

class BookSearchOrchestrator:
    def __init__(self, bookstore_urls: List[str],
                  max_workers: int = 50):
        self.clients = [
            BookstoreClient(url)
            for url in bookstore_urls
        ]
        self.max_workers = max_workers

    def find_lowest_price(
        self, title: str,
        timeout: float = 15.0
    ) -> Optional[float]:
        """
        Query ALL bookstores in parallel.
        Return lowest price found within timeout.
        Key: ThreadPoolExecutor for parallel I/O!
        """
        prices = []
        start = time.time()

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    client.search, title
                ): client
                for client in self.clients
            }
            for future in as_completed(
                futures,
                timeout=timeout - (
                    time.time() - start
                )
            ):
                try:
                    price = future.result()
                    if price is not None:
                        prices.append(price)
                except Exception:
                    pass

        return min(prices) if prices else None

class PaymentService:
    def charge(self, cc: str,
                amount: float) -> dict:
        # In production: Stripe/payment gateway
        return {
            "status": "success",
            "amount": amount,
            "transaction_id": "txn_abc123"
        }

class BookService:
    def __init__(self, seller_urls: List[str]):
        self.orchestrator = \
            BookSearchOrchestrator(seller_urls)
        self.payment = PaymentService()

    def process_request(
        self, title: str,
        cc: str, max_price: float
    ) -> dict:
        start = time.time()

        # Step 1: search all bookstores
        lowest = self.orchestrator\
            .find_lowest_price(title)

        elapsed = time.time() - start

        if lowest is None:
            return {
                "status": "not_found",
                "message": f"'{title}' unavailable",
                "search_time": elapsed
            }

        # Step 2: price decision
        if lowest <= max_price:
            # Step 3: charge card
            payment = self.payment.charge(
                cc, lowest
            )
            return {
                "status": "purchased",
                "price": lowest,
                "payment": payment,
                "search_time": elapsed
            }
        else:
            return {
                "status": "too_expensive",
                "lowest_available": lowest,
                "your_max": max_price,
                "search_time": elapsed
            }

# Test with 20 simulated bookstores
sellers = [f"https://store{i}.com"
            for i in range(20)]
service = BookService(sellers)

print("Testing Bookstore Service...")
result = service.process_request(
    title="Clean Code",
    cc="4242424242424242",
    max_price=30.0
)
print(f"Result: {result['status']}")
print(f"Search time: {result['search_time']:.2f}s")
if 'price' in result:
    print(f"Price: ${result['price']:.2f}")
elif 'lowest_available' in result:
    print(f"Lowest found: "
          f"${result['lowest_available']:.2f}")
```

## SYSTEM 2: Throttle System
(Confirmed asked in 2026 Databricks interviews!)

### Requirements
Client → HTTP Server → API Server → DB/3rd Party
- Internal + external users
- Traffic bursts cause cascading failures
- Design throttling to prevent overloads

### Architecture Decisions
Client → API Gateway
↓
[Rate Limiter Layer]
Sliding window per user/IP
Token bucket per service
↓
[Circuit Breaker]
CLOSED → OPEN → HALF-OPEN
Opens when error rate > 50%
↓
API Server
       ↓
[Backpressure + Queue]
Reject when queue full
Priority queues (internal > external)
  ↓
DB / 3rd Party

### Key Patterns

**Circuit Breaker States:**
CLOSED (normal):  requests pass through
track error rate
OPEN (failing):   reject ALL requests fast
wait N seconds
HALF-OPEN:        let 1 request through
if success → CLOSED
if fail → OPEN again

**Priority Queue:**
```python
# Internal teams get higher priority
PRIORITIES = {
    "internal": 0,   # highest
    "premium": 1,
    "standard": 2,
    "free": 3        # lowest
}
import heapq
queue = []
heapq.heappush(queue,
    (PRIORITIES["internal"], request))
```

## SYSTEM 3: ML Model Serving at Scale

### Requirements
- Serve 1000+ RPS with <100ms p99
- Multiple model versions
- A/B traffic splitting
- Auto-scaling
- Zero downtime deploys

### Architecture
Load Balancer (NGINX)
↓
API Gateway (FastAPI, 3 replicas)
↓
Model Router (traffic split)
├── 80% → Model v1 (stable)
└── 20% → Model v2 (candidate)
↓
Model Servers (Docker, auto-scale 2-10)
↓
Feature Store (Redis, <5ms lookup)
↓
Monitoring (Prometheus + Grafana)

**Scale decisions:**
- Horizontal scaling: add servers (not bigger)
- Feature caching: Redis for hot features
- Model caching: keep in memory, reload on update
- Health checks: /health every 10s
- Rollback: keep v1 running during v2 test
