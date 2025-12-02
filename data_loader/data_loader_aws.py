import os
import s3fs
import polars as pl


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

    def save_parquet(self, data: pl.DataFrame, relative_path: str):
        """
        save Parquet file polars dataframe
        """
        FILE_PATH_S3 = self.bucket + "/" + relative_path

        with self.fs.open(FILE_PATH_S3, mode="w") as file_out:
            data.write_parquet(file_out)
