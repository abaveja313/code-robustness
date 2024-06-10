import threading
from contextlib import contextmanager
from hashlib import sha1

from cachetools import LRUCache
from loguru import logger


class LongMessageHashFilter:
    def __init__(self, min_length, max_cache_size):
        self.min_length = min_length
        self.cache = LRUCache(maxsize=max_cache_size)
        self.lock = threading.Lock()

    def __call__(self, record):
        record['extra']['hash'] = ''  # Default
        log_message = record["message"]

        if len(log_message) < self.min_length:
            return True  # Allow the message to be logged as usual

        message_hash = sha1(log_message.encode()).hexdigest()

        with self.lock:
            cached_message = self.cache.get(message_hash)

            if cached_message:
                record["message"] = f"Cached message: See hash {message_hash}"
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
