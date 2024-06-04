from gcsfs import GCSFileSystem
import json

from shared.structs import BenchmarkResult
from loguru import logger


class GCSResultStorageManager:
    RESULT_PATH = "results.json"

    def __init__(
            self,
            model_name: str,
            bucket_name: str = "amrit-research-samples",
            project: str = "research",
            service_account_file: str = "/home/user/service-account.json",
    ):
        self.model_name = model_name.replace('/', '_')
        self.bucket_name = bucket_name
        self.gcs = GCSFileSystem(project=project, token=service_account_file)

    def add(self, result: BenchmarkResult):
        logger.info(f"Adding result to GCS")
        json_line = json.dumps(result.__dict__) + "\n"

        full_path = f"{self.bucket_name}/{self.model_name}/{result.problem_id}_{result.mutation}_{result.stem_id}.jsonl"

        with self.gcs.open(full_path, "w") as f:
            f.write(json_line)
