import polars as pl


def clean_na_values(data: pl.DataFrame, threshold: float = 0.01) -> pl.DataFrame:
    """
    Clean DataFrame by removing columns exceeding a null-value ratio.
    """
    null_ratio = data.null_count() / data.height

    cols_to_keep = [c for c in data.columns if null_ratio[c].item() <= threshold]

    return data.select(cols_to_keep).drop_nulls()


def select_symbol_id(data: pl.DataFrame, id: int = 1) -> pl.DataFrame:
    """
    Filter for a given symbol_id.
    """

    data_filter_symb = data.filter(pl.col("symbol_id") == id)
    data_filter_symb = data_filter_symb.sort(["date_id", "time_id"])

    return data_filter_symb
