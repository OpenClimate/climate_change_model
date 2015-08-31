""" Module to load various historical datasets for model comparison.
"""
import pandas as pd
import numpy as np
from SoCCo.data.local import mauna_loa_co2_filepath, climate_dat_filepath
from SoCCo.data.web import giss_temp_url
from SoCCo.utils.pandas_utils import unstack_and_build_index


def load_global_co2(source="Mauna Loa", frequency="Y"):
    """ Global atmospheric CO2 concentrations measured by various sources.

    Concentrations are in part per million (ppm).

    Returns
    -------
    co2_data : pandas.Series
        Series with index as a period containing the CO2 atmospheric
        concentrations in ppm.
    """
    if source == "Mauna Loa":
        return load_mauna_loa_co2(frequency=frequency)
    elif source == "climateDat":
        return load_mauna_loa_co2(frequency=frequency)
    else:
        raise NotImplementedError("Global CO2 cannot be loaded from {}. "
                                  "Currently only supporting 'Mauna Loa' and "
                                  "'climateDat'.")


def load_mauna_loa_co2(frequency="Y"):
    """ Global atmospheric CO2 concentrations measured by Mauna Loa observat.

    Concentrations are in part per million (ppm).

    Returns
    -------
    co2_data : pandas.Series
        Series with index as a period containing the CO2 atmospheric
        concentrations in ppm.
    """
    co2_data=pd.read_excel(mauna_loa_co2_filepath,
                           sheetname='Monthly & Annual CO2 Data', skiprows=6)
    co2_data = co2_data.set_index('Year')
    if frequency == "Y":
        co2_data = co2_data[u'Annual Average']
        co2_data.index = pd.to_datetime(co2_data.index, format="%Y")
        co2_data.index = co2_data.index.to_period()
    elif frequency == "M":
        co2_data = co2_data.loc[:, :'Dec']
        co2_data = unstack_and_build_index(co2_data)
    else:
        raise NotImplementedError("Unsupported frequency. Please select 'Y' "
                                  "(year) or 'M' (month).")
    co2_data.name = "Co2 (ppm)"
    return co2_data


def load_climateDat_co2(frequency="Y"):
    """ Global atmospheric CO2 concentrations measured by ???.

    Concentrations are in part per million (ppm).

    Returns
    -------
    co2_data : pandas.Series
        Series with index as a period containing the CO2 atmospheric
        concentrations in ppm.
    """
    co2Data = pd.read_csv(climate_dat_filepath)
    return co2Data


def load_global_temperature(source="GISS", frequency="Y"):
    """ Global land-ocean montly and yearly temperature changes in deg Celsius.

    Returns
    -------
    temp : pandas.Series
        Series with index as a period containing the temperature changes in
        degC.
    """
    if source == "GISS":
        return load_giss_t(frequency=frequency)
    else:
        raise NotImplementedError("Global CO2 cannot be loaded from {}. "
                                  "Currently only supporting 'Mauna Loa' and "
                                  "'climateDat'.")


def load_giss_t(frequency="Y"):
    """ Global land-ocean montly and yearly temperature changes in deg Celsius.

    The changes are with respect to the period of 1951-1980. Absolute yearly
    mean estimate over base period is 14 degC.

    Returns
    -------
    temp : pandas.Series
        Series with index as a period containing the temperature changes in
        degC.
    """
    giss_temp = pd.read_table(giss_temp_url, sep="\s+", skiprows=7,
                              skip_footer=11, engine="python")
    giss_temp = giss_temp.set_index('Year')
    # Clean up
    giss_temp = giss_temp.drop("Year")
    giss_temp = giss_temp.where(giss_temp != "****", np.nan)
    giss_temp = giss_temp.where(giss_temp != "***", np.nan)
    if frequency == "Y":
        temp = giss_temp['J-D']
        temp.index = pd.to_datetime(temp.index, format="%Y").to_period()

    elif frequency == "M":
        temp = giss_temp.loc[:, 'Jan':'Dec']
        temp = unstack_and_build_index(temp)

    temp.name = 'Global Temperature change (degC)'
    temp = temp.astype(np.float64) / 100. # Convert to changes in deg C.
    return temp


if __name__ == "__main__":
    co2_monthly = load_global_co2(frequency="M")
    co2_yearly = load_global_co2(frequency="Y")

    t_yearly = load_global_temperature(frequency="Y")
    t_monthly = load_global_temperature(frequency="M")
