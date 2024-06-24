import random
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from hashlib import md5

from cachetools import TTLCache
from loguru import logger


def prob_log(message: str, p=0.05):
    if random.random() < p:
        logger.info(message)


@contextmanager
def log_time(op_name="Operation"):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"{op_name} - Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.2f} milliseconds)"
        )


class LongMessageHashFilter:
    def __init__(self, min_length, max_cache_size, ttl: timedelta):
        self.min_length = min_length
        self.cache = TTLCache(maxsize=max_cache_size, ttl=ttl.total_seconds())
        self.lock = threading.Lock()

    def __call__(self, record):
        record["extra"]["hash"] = ""  # Default
        log_message = record["message"]

        if len(log_message) < self.min_length:
            return True  # Allow the message to be logged as usual

        message_hash = md5(log_message.encode()).hexdigest()

        with self.lock:
            cached_message = self.cache.get(message_hash)

            if cached_message:
                record["message"] = f"Cached message"
                record["extra"]["hash"] = message_hash
            else:
                record["extra"]["hash"] = message_hash
                self.cache[message_hash] = log_message

        return True


@contextmanager
def create_problem_logger(problem_id: str, rotation="10 MB", level="DEBUG"):
    logger_id = None
    try:
        formatted_problem_id = problem_id.replace("/", "_").lower()
        log_file = f"logs/{formatted_problem_id}.log"
        logger_id = logger.add(
            log_file, compression="zip", rotation=rotation, level=level
        )
        yield
    finally:
        logger.remove(logger_id)
