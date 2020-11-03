import sys
import numpy as np
import pandas as pd

koidata = "cumulative.csv"

df = pd.read_csv(koidata, header = 0)
original_headers = list(df.columns.values)
df = df._get_numeric_data()
numeric_headers = list(df.columns.values)
numpy_array = df.to_numpy()
numeric_headers.reverse()
reverse_df = df[numeric_headers]
reverse_df.to_excel('koinumpy.xls')