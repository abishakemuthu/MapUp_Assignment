import pandas as pd
import numpy as np
from datetime import time, timedelta

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here  
    toll_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    toll_ids = sorted(toll_ids)
    
    n = len(toll_ids)
    distance_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)
    
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance
    
    # Floyd-Warshall algorithm
    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    results = []
    
    ids = df.index.tolist()
    
    for id_start in ids:
        for id_end in ids:
            if id_start != id_end:
                distance = df.loc[id_start, id_end]
                results.append((id_start, id_end, distance))
    
    return pd.DataFrame(results, columns=['id_start', 'id_end', 'distance'])


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    if reference_id not in df['id_start'].values:
        raise ValueError("Reference ID not found in the DataFrame.")
    
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    average_distance = reference_distances.mean()
    
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    filtered_ids = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]['id_start']
    
    return sorted(filtered_ids.unique())


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle_type, rate in rates.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    weekdays_discount = {
        'morning': 0.8,
        'day': 1.2,
        'evening': 0.8
    }
    
    weekend_discount = 0.7
    
    df['start_day'] = 'Monday'
    df['end_day'] = 'Monday'
    
    df['start_time'] = time(0, 0)
    df['end_time'] = time(23, 59)
    
    for index, row in df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        start_day = row['start_day']
        end_day = row['end_day']
        
        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if start_time < time(10, 0):
                discount_factor = weekdays_discount['morning']
            elif start_time < time(18, 0):
                discount_factor = weekdays_discount['day']
            else:
                discount_factor = weekdays_discount['evening']
        else:
            discount_factor = weekend_discount
        
        df.at[index, 'moto'] *= discount_factor
        df.at[index, 'car'] *= discount_factor
        df.at[index, 'rv'] *= discount_factor
        df.at[index, 'bus'] *= discount_factor
        df.at[index, 'truck'] *= discount_factor

    return df
