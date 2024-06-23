import json
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas
import pandas as pd
import tqdm
from gcsfs import GCSFileSystem
import sys
import math

sys.path.append("../src")


class SampleAggregator:
    def __init__(self,
                 project: str = "research",
                 service_account_file: str = "/Users/amrit/Downloads/service-account.json"):
        self.project = project
        self.service_account_file = service_account_file
        self.gcs = GCSFileSystem(project=project, token=self.service_account_file)

    @staticmethod
    def safe_log_ratio(mutated, original, epsilon=1e-10):
        if original == 0 and mutated == 0:
            return 0.0  # No change
        elif original == 0:
            return math.log(mutated / epsilon)  # Large positive value
        elif mutated == 0:
            return -math.log(original / epsilon)  # Large negative value
        else:
            return math.log(mutated / original)

    @staticmethod
    def symmetric_percent_change(mutated, original):
        if mutated == original:
            return 0
        return 200 * (mutated - original) / (mutated + original)

    @staticmethod
    def percent_change(mutated, original):
        if mutated == original:
            return 0
        if original == 0:
            return float('inf')
        return 100 * (mutated - original) / original

    def aggregate(self, model_name: str):
        df = []
        targets = [t for t in self.gcs.find(f"gcs://amrit-research-samples/{model_name}") if t.endswith('.jsonl')]

        def read_file(target):
            with self.gcs.open(target) as f:
                return list(map(json.loads, f))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(read_file, target) for target in targets]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Reading Data"):
                df.extend(future.result())

        for row in tqdm.tqdm(df, desc="Postprocessing Data"):
            # Expand nested columns
            for pass_at in ['pass_at_original', 'pass_at_mutated']:
                prefix, suffix = pass_at.rsplit('_', 1)

                for col in row[pass_at]:
                    # pass_at_1_original
                    row[f"{prefix}_{col}_{suffix}"] = row[pass_at][col]

            for k in row['pass_at_ratio']:
                row[f"pass_at_{k}_diff"] = row['pass_at_diff'][k]
                row[f"pass_at_{k}_ratio"] = row['pass_at_ratio'][k]
                row[f"pass_at_{k}_log_ratio"] = self.safe_log_ratio(row[f'pass_at_{k}_mutated'],
                                                                    row[f'pass_at_{k}_original'])
                row[f"pass_at_{k}_sym_percent_change"] = self.symmetric_percent_change(row[f'pass_at_{k}_mutated'],
                                                                                       row[f'pass_at_{k}_original'])
                row[f"pass_at_{k}_percent_change"] = self.percent_change(row[f'pass_at_{k}_mutated'],
                                                                         row[f'pass_at_{k}_original'])
        return pd.DataFrame(df)
