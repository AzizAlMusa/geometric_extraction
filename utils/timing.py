# timing.py
# utils/timing.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name="Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end-start:.3f}s")
