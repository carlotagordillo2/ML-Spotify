import numpy as np

def remove_outliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    
    # Filter the DataFrame based on the condition
    filtered_data = data[(data[col] > lower_bound) & (data[col] < upper_bound)]

    return filtered_data


def percentage_outliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #print("Lower Bound:", lower_bound)
    #print("Upper Bound:", upper_bound)
    
    # Filter the DataFrame based on the condition
    filtered_data = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

    
    percentage_outliers = (len(filtered_data) / len(data))*100
    
    percentage = f'The percentage of outliers in {col} is {percentage_outliers}\n'
    
    print(percentage)

    return percentage_outliers

def create_frequency_table(df,name):
    
    # control
    frequency_table_control = df[name].value_counts()
   
    proportion_table_control = df[name].value_counts(normalize=True)
    
    
    return frequency_table_control, proportion_table_control