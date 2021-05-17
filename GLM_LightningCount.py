import xarray as xr
import pandas as pd
import numpy as np
import os
import csv
import logging


'''
Outputs lightning count for every file in the processed GLM directory
glm_list: your directory of processed GLM files, with one file per day
'''

glm_dict = {}
print(type(glm_dict))
glm_list = os.listdir('/glade/scratch/gwallach/goes16_nc/GLM_grids')
#initilize the different columns
print(glm_list)
counter = 0
final_sum = 0
for glm_test in glm_list:
    file_sum = 0
    glm = xr.open_dataset('/glade/scratch/gwallach/goes16_nc/GLM_grids/' + glm_test)
    #print(glm)
    #use np.sum() to get lightning counts
    #you can also use np.count_nonzero()
    for i in glm.data_vars['lightning_counts']:
        #print(i)
        for j in i:
            #print(j)
            for k in j:
                #print(k)
                file_sum = file_sum + 1
                if k.values > 0.0:
                    #print("Found One!", k.values)
                    counter = counter + 1
                    #print("Counter = ", counter)
    #print("Total Counter =", counter)
    #look at how the append command works
    #each row should be a different date, with the differnt 
    #create dictionary of lists 
    #use pandas concate
    final_sum = final_sum + file_sum
    #print('Dataframe =', glm_dataframe)
df = pd.DataFrame({"Lightning Strikes": counter, "Full Total": final_sum})    
glm_dataframe = xr.DataArray(df)
glm_dataframe.to_csv('GLM_LigtningCount.csv',index=False)
print(glm_dataframe)