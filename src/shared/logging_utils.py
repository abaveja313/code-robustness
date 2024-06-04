from loguru import logger
from contextlib import contextmanager


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
