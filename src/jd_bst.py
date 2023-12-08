import numpy as np
from numpy import diff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from IPython.display import clear_output
import scipy
import os 
print(os.getcwd())
from tqdm import tqdm

os.chdir('src')
df_mens = pd.read_csv('svk_height_weight_mens_2008_v2.csv',sep=',', index_col=0).reset_index(drop=True)
atv_column_names = [x for x in df_mens.columns if 'ATV' in x]
df_mens_atv = df_mens[atv_column_names]

att_column_names = [x for x in df_mens.columns if 'ATT' in x]
df_mens_att = df_mens[att_column_names]
atv_column_names_gradient = []
for i in range(8, 18):
    atv_column_names_gradient.append('ATV_diff_'+str(i)+'-'+str(i+1))
#Height men
number_of_similar_rows_to_include_for_predictions = 100
all_all_errors=[]
from numpy import diff
for start_age in tqdm(range(8,9)): #Predpostavimo da imamo vsa leta za nazaj, ker razlike niso tako velike
    print('------------------')
    print('start_age: ' + str(start_age))
    print('------------------')
    col_index_for_start_age = start_age-8 #first year = 8
    diffs = diff(df_mens_atv[df_mens_atv.columns[col_index_for_start_age:]].values)
    atv_column_names_gradient_current = atv_column_names_gradient[col_index_for_start_age:]
    df_grad = pd.DataFrame(data=diffs, columns=atv_column_names_gradient_current)

    all_errors=[]
    all_std_errors = []
    for age in tqdm(range(start_age,18)):
        print('Data up to age: ' + str(age))
        col_index_for_age = age-start_age # start_age = 8
        
        #remaining_ages_for_difference = 18-age
        
        #Calculate similarities between year differences
        #Simialrities are calculated only for values lower than age
        similarity = scipy.spatial.distance.cdist(df_mens_atv[df_mens_atv.columns[col_index_for_start_age:col_index_for_start_age + col_index_for_age+1]], df_mens_atv[df_mens_atv.columns[col_index_for_start_age:col_index_for_start_age + col_index_for_age+1]], metric='euclidean')

        predicting_errors =[]
        all_errors=[]
        all_std_errors = []
        all_heights = []
        for remaining_ages_for_difference in tqdm(range(0, 18-age+1)): #, 18-age+1):
        #for remaining_ages_for_difference in range(1, 2): #, 18-age+1):
            print(f'Predicting for {remaining_ages_for_difference} years ahead')
            #print(remaining_ages_for_difference)
            predicting_errors =[]
            #We iterate through rows
            for i in range(len(similarity)):
                        
                row = similarity[i]
                #print(df_mens_atv.iloc[i][df_mens_atv.columns[col_index_for_start_age + col_index_for_age]])
                #print(df_mens_atv.iloc[i][df_mens_atv.columns[col_index_for_start_age + col_index_for_age + remaining_ages_for_difference]])
                sorted_indexes = list(np.argsort(row)) #We sort values from biggest similarity to lowest-we get indexes of those values
                del sorted_indexes[sorted_indexes.index(i)] #Remove value of index of the same value to remove similati of the same values

                #We get rows with simmilar differences
                similar_rows = []
                for j in range(number_of_similar_rows_to_include_for_predictions):
                    similar_rows.append(df_grad.iloc[sorted_indexes[j]]) #Original version                    

                #print(similar_rows[0:3])
                #print('-------------------')
                
                #Now we need to calculate remaining differences to estimate value at x
                next_diferences = [0]*remaining_ages_for_difference
                #print(similar_rows[-1])
                #print()
                for similar_row in similar_rows:
                    for k in range(1, remaining_ages_for_difference+1):
                        #print('k')
                        #print(col_index_for_start_age + col_index_for_age)
                        #print(k-1)
                        #print(similar_row[k-1])
                        #next_diferences[-k-1] = next_diferences[-k-1] + similar_row[-k-1] #we add difference from simmilar rows for each year
                        next_diferences[k-1] = next_diferences[k-1] + similar_row[col_index_for_start_age + col_index_for_age + k-1] #we add difference from simmilar rows for each year
                        #next_diferences[k] = next_diferences[k] + similar_row[k]
                        #print(f'Similar row value: {similar_row[k]}')
                        
                #print(next_diferences)
                next_diferences = np.array(next_diferences)/number_of_similar_rows_to_include_for_predictions #divide to get average
                #print('next')
                #print(next_diferences)
                #break
                all_next_differences = sum(next_diferences)
                current_value = df_mens_atv.iloc[i][df_mens_atv.columns[col_index_for_start_age + col_index_for_age]]
                
                predicted_value = current_value + all_next_differences
                #actual_value = df_mens_atv.iloc[i][df_mens_atv.columns[-1]]
                actual_value = df_mens_atv.iloc[i][df_mens_atv.columns[col_index_for_start_age + col_index_for_age + remaining_ages_for_difference]]
                #print(actual_value)
                predicting_error = abs(predicted_value - actual_value)
                predicting_errors.append(predicting_error)

            print('done')    
            #Average predicting error
            predicting_errors = [x for x in predicting_errors if x < 150]
            average_error = np.median(predicting_errors)
            std_error = np.std(predicting_errors)
            print('Average error: ' + str(average_error))
            #print(predicting_errors)
            print('Std. error: ' + str(std_error))
            #print predicted value
            print('Predicted Height: ' + str(predicted_value))
            all_errors.append(average_error)
            all_std_errors.append(std_error)
            all_heights.append(predicted_value)
            
        print(all_errors)
        print(all_std_errors)
        print(all_heights)
        print('Sum errors: ' + str(sum(all_errors)))

        pd.concat([all_errors, all_heights, all_std_errors])