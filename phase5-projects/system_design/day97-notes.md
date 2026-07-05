## Bookstore System Key Decisions
Parallel I/O: ThreadPoolExecutor (not async!)
  → Bookstore APIs are blocking → threads win
  → async better for non-blocking I/O

Timeout: 15s total, pass to as_completed()
  → Never wait for slowest seller forever!

Payment: only charge if found AND affordable
  → Always validate before charging!

Scale: cache popular books (Redis, 1hr TTL)
       queue requests (SQS) during spikes

## Circuit Breaker Pattern
CLOSED → OPEN: error rate > 50% in window
OPEN → HALF-OPEN: after timeout (30s)
HALF-OPEN → CLOSED: probe request succeeds
HALF-OPEN → OPEN: probe request fails
Use: wrap ALL external API calls!

## STAR Story Framework
Situation: set the scene (1-2 sentences)
Task:      your responsibility (1 sentence)
Action:    what YOU did (3-4 bullet points)
Result:    quantified outcome (metrics!)
Always end with: what you learned

## Multi-source BFS Template
Initialize queue with ALL sources
Track time/level per element
Process layer by layer
Return max level when queue empty
