from os.path import abspath, dirname, join

HERE = dirname(__file__)

mauna_loa_co2_filename = "co2-atmospheric-mlo-monthly-scripps.xls"
mauna_loa_co2_filepath = abspath(join(HERE, mauna_loa_co2_filename))

climate_dat_filename = "climateDat.csv"
climate_dat_filepath = abspath(join(HERE, climate_dat_filename))
