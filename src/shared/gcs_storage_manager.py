import pickle
from typing import Any

from gcsfs import GCSFileSystem
import json

from shared.structs import BenchmarkResult
from loguru import logger
import pickle


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

    def add_all(self, results: dict[str, BenchmarkResult]):
        for result in results:
            self.add(results[result])

    def get_data_pickles(self):
        full_path = f"{self.bucket_name}/{self.model_name}/pickles/"

        for file in self.gcs.ls(full_path):
            if file.endswith(".pkl"):
                with self.gcs.open(file, "rb") as f:
                    obj = pickle.loads(f.read())
                    if len(obj['results']) == 0:
                        raise ValueError("No results found in pickle")

                    fres: BenchmarkResult = list(obj['results'].values())[0]
                    yield fres.problem_id, obj['evaluate_targets'], obj['results']

    def add_data_pickle(self, eval_target: dict[str, Any], problem_id: str):
        logger.info(f"Adding pickle to GCS")
        problem_id = problem_id.replace("/", "_").lower()
        full_path = f"{self.bucket_name}/{self.model_name}/pickles/problem_{problem_id}.pkl"

        with self.gcs.open(full_path, "wb") as f:
            f.write(pickle.dumps(eval_target))

    def add(self, result: BenchmarkResult):
        logger.info(f"Adding result to GCS")
        json_line = json.dumps(result.__dict__) + "\n"

        full_path = (f"{self.bucket_name}/{self.model_name}/{result.problem_id}/temp_{result.temp}/{result.mutation}"
                     f"_{result.stem_id}_{result.temp}.jsonl")

        with self.gcs.open(full_path, "w") as f:
            f.write(json_line)
