# src/benchmark.py
import time
from src.infer import predict
texts = ["Starbucks 123"]*1000
# single
t0 = time.time()
for x in texts:
    predict([x])
t1 = time.time()
print("Single-call avg (s):", (t1-t0)/len(texts))
# batch
t0 = time.time()
predict(texts)
t1 = time.time()
print("Batch avg (s):", (t1-t0)/len(texts))
