import os
import pandas as pd


def pd_xlsx_write(df: pd.DataFrame, path: str, sheet_name: str, append: bool = True):
    """
    Writes dataframe to path
    args:
        df:             data frame
        path:           path to write/append
        sheet_name:     name of excel sheet
        append:         whether to append data if the file already exists
    """
    writing_mode = "a" if os.path.isfile(path) and append else "w"
    with pd.ExcelWriter(path, mode=writing_mode) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
