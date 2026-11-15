from slowapi import Limiter
from slowapi.util import get_remote_address

# Counts requests per client IP. Stored in memory, which is fine with one worker.
# Switch to Redis (storage_uri=...) before running multiple workers.
limiter = Limiter(key_func=get_remote_address)