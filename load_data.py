# %%
import os
import polars as pl
import time
from tqdm import tqdm
from config import USER

from data_loader import (
    clean_na_values,
    select_symbol_id,
    S3ParquetReader,
)

# This code make the assumption the data are stored in https://datalab.sspcloud.fr/file-explorer/[user]/
BUCKET = f"/{USER}/jane_street_data"


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
data_partitions = []
t0 = time.time()
for i in tqdm(range(10), desc="Partition_id"):
    FILE_KEY_S3 = f"train.parquet/partition_id={i}/part-0.parquet"
    data_partitions.append(reader.read_parquet(FILE_KEY_S3))

delta_t = time.time() - t0
delta_t
# %%
data = pl.concat(data_partitions)
data_clean_symb_1 = data.pipe(clean_na_values, threshold=0.05).pipe(
    select_symbol_id, id=1
)
data_clean_symb_1.head()
