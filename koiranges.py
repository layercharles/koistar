# what is the range of values of both stellar effective temperature and planttary radius that the kepler program would deem a confirmed exoplanet
# data we are using 
#     identifiers
#         rowid
#         kepid
#         kepoi_name
#         kepler_name
#         koi_disposition - confirmed
# data we are comparing to identifiers - koi_disposition - confirmed
#     range1
#         koi_impact
#         koi_impact_err1
#         koi_impact_err2
#     range2
#         koi_prad
#         koi_prad_err1
#         koi_prad_err2
#     range3
#         koi_steff
#         koi_steff_err1
#         koi_steff_err2
#     range4
#         koi_depth
#         koi_depth_err1
#         koi_depth_err2
# import os
import sys
import numpy as np
import pandas as pd


koic = pd.read_csv('cumulative.csv', header=0)
#koidf = pd.DataFrame(koic, columns=["rowid", "kepid", "kepoi_name", "kepler_name", "koi_disposition", 
                        #  "koi_impact", "koi_impact_err1", "koi_impact_err2", "koi_prad", "koi_prad_err1", "koi_prad_err2", "koi_steff", "koi_steff_err1", "koi_steff_err2", "koi_depth", "koi_depth_err1", "koi_depth_err2"])
# fname = os.path.join(r'C:\Users\Charles\Desktop\python\archive\cumulative.csv')
# sp = koidf.at[1, 'koi_disposition']
# print(sp)
# dfm = koidf['koi_disposition']=='CONFIRMED'
# filtered_koidf = koidf[dfm]
# print(filtered_koidf['koi_steff'].mean())
# filtered_koidf.to_csv('out.csv')
# print(koidf.head())
# print(koidf.head())




