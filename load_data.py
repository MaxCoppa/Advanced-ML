# %%
import os
import s3fs
import polars as pl
import time
from tqdm import tqdm
from config import USER

# This code make the assumption the data are stored in https://datalab.sspcloud.fr/file-explorer/[user]/
BUCKET = f"/{USER}/jane_street_data"


class S3ParquetReader:
    def __init__(self, bucket: str):
        self.bucket = bucket

        # EXACT initialization s3fs https://docs.sspcloud.fr/content/storage.html
        S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
        self.fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    def read_parquet(self, relative_path: str) -> pl.DataFrame:
        """
        read Parquet file polars dataframe
        """
        FILE_PATH_S3 = self.bucket + "/" + relative_path

        with self.fs.open(FILE_PATH_S3, mode="rb") as file_in:
            return pl.read_parquet(file_in)


# %%


reader = S3ParquetReader(bucket=BUCKET)

# %%
FILE_KEY_S3 = "train.parquet/partition_id=0/part-0.parquet"
df_bpe = reader.read_parquet(FILE_KEY_S3)
df_bpe.shape
# %%
all_buckets = reader.fs.ls(BUCKET + "/" + "train.parquet/")
all_buckets
# %%
# Load all files
t0 = time.time()
for i in tqdm(range(10), desc="Partition_id"):
    FILE_KEY_S3 = f"train.parquet/partition_id={i}/part-0.parquet"
    reader.read_parquet(FILE_KEY_S3)

delta_t = time.time() - t0
# %%
delta_t
# %%

