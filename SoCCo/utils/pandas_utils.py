""" Dataframe manipulation utilities.
"""
import pandas as pd


def unstack_and_build_index(df):
    """ Convert a dataframe with years along index and months along columns and
    build a correctly index timeseries, with monthly periods.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of the form:
              Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
        1980    0   1   2   3   4   5   6   7   8   9  10  11
        1981   12  13  14  15  16  17  18  19  20  21  22  23

    Returns
    -------
    df : pandas.Series
        Series index with a monthly period of the form:
        1980-01 0
        1980-02 1
        ...
    """
    series_out = df.transpose().unstack()
    series_out.name = "Values"
    series_out.index.names = ["Year", "Month"]
    df_out = series_out.reset_index()
    df_out.Year = df_out.Year.astype("str")
    # Temporarily set it to the first of the month since to_datetime fails for
    # Feb otherwise
    df_out["str_index"] = "01/" + df_out.Month + "/" + df_out.Year
    df_out.index = df_out["str_index"].apply(pd.to_datetime)
    df_out.index = df_out.index.to_period()
    df_out.index.name = "Date"
    return df_out["Values"]
