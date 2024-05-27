from gcsfs import GCSFileSystem
import json

from shared.structs import BenchmarkResult


class GCSResultStorageManager:
    RESULT_PATH = "results.json"

    def __init__(
        self,
        bucket_name: str = "amrit-research-samples",
        project: str = "research",
        service_account_file: str = "/home/user/service-account.json",
    ):
        self.bucket_name = bucket_name
        self.gcs = GCSFileSystem(project=project, token=service_account_file)

    def add(self, result: BenchmarkResult):
        json_line = json.dumps(result.__dict__) + "\n"

        full_path = f"{self.bucket_name}/{GCSResultStorageManager.RESULT_PATH}"

        try:
            with self.gcs.open(full_path, "r") as f:
                existing_content = f.read()
        except FileNotFoundError:
            existing_content = ""

        updated_content = existing_content + json_line

        with self.gcs.open(full_path, "w") as f:
            f.write(updated_content)
