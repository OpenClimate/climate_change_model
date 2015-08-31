""" Demo script for loading and plotting reference dataasets.
"""

from SoCCo.data.data_loaders import load_global_co2, load_global_temperature
from matplotlib.pyplot import figure, show, subplot

co2_monthly = load_global_co2(frequency="M")
co2_yearly = load_global_co2(frequency="Y")
figure()
ax = subplot(1, 1, 1)
co2_monthly.plot(ax=ax)
co2_yearly.plot(ax=ax)

t_yearly = load_global_temperature(frequency="Y")
t_monthly = load_global_temperature(frequency="M")
figure()
ax = subplot(1, 1, 1)
t_yearly.plot(ax=ax)
t_monthly.plot(ax=ax)
show()
