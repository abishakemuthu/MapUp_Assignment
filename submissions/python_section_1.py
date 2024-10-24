from typing import Dict, List
import pandas as pd
import re
import polyline
from datetime import datetime, timedelta


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    main_list = []
    for i in range(0,len(lst),n):
        sub_list = lst[i:i+n]

        rev_list = []
        for j in range(len(sub_list)):
            rev_list.insert(0,sub_list[j])
        
        main_list.extend(rev_list)

    return main_list


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    d = {}
    for i in lst:
        if len(i) not in d.keys():
            d[len(i)]=[i]
        else:
            d[len(i)].append(i)

    return dict(sorted(d.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    d = {}

    for k,v in nested_dict.items():
        if type(v) == dict:
            sub_dict = flatten_dict(v, sep)
            for k1,v1 in sub_dict.items():
                d[f"{k}{sep}{k1}"] = v1
        elif type(v) == list:
            for i, j in enumerate(v):
                if type(j) == dict:
                    sub_dict = flatten_dict(j, sep)
                    for k3,v3 in sub_dict.items():
                        d[f"{k}[{i}]{sep}{k3}"] = v3
                else:
                    d[f"{k}[{i}]"] = j
        else:
            d[k] = v

    return d

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    l = []
    for i in range(len(nums)):
        x = nums.pop(-1)
        nums.insert(0,x)
        l.append(nums.copy())

    return l


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    return re.findall(r'\d+[./-][\d]+[./-]*[\d]*',text)

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of the Earth in meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])  # Convert degrees to radians
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    df['distance'] = 0.0
    
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df


def rotate_and_multiply_matrix(matrix: list[list[int]]) -> list[list[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    matrix = matrix[::-1]
    print("Original: ",matrix)
    m = []
    for i in range(len(matrix)):
        l = []
        for j in matrix:
            l.append(j[i])
        m.append(l)

    print("Rotated: ",m)

    z = [[0] * len(m) for i in range(len(m))]

    for i in range(len(m)):
        for j in range(len(m)):
            row = sum(m[i])-m[i][j]
            col = sum(row[j] for row in m)-m[i][j]
            z[i][j] = row+col

    print("Final: ",z)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    def get_datetime(day, time):
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_index = days_of_week.index(day)
        return datetime(2024, 1, 1 + day_index) + pd.to_timedelta(time)
    
    df['start_datetime'] = df.apply(lambda row: get_datetime(row['startDay'], row['startTime']), axis=1)
    df['end_datetime'] = df.apply(lambda row: get_datetime(row['endDay'], row['endTime']), axis=1)

    full_week_start = datetime(2024, 1, 1, 0, 0, 0)  # Monday 00:00:00
    full_week_end = datetime(2024, 1, 7, 23, 59, 59) # Sunday 23:59:59

    df['id'] = df['id'].astype(str)
    df['id_2'] = df['id_2'].astype(str)

    completeness_check = pd.Series(dtype=bool, index=pd.Index(df[['id', 'id_2']].apply(tuple, axis=1).drop_duplicates(), name='pair'))

    for (id_, id_2), group in df.groupby(['id', 'id_2']):
        sorted_intervals = group[['start_datetime', 'end_datetime']].sort_values(by='start_datetime')
        
        current_time = full_week_start
        
        complete = True
        for _, row in sorted_intervals.iterrows():
            if row['start_datetime'] > current_time:
                complete = False
                break
            current_time = max(current_time, row['end_datetime'])
        
        if current_time < full_week_end:
            complete = False
        
        completeness_check[(id_, id_2)] = not complete
    
    return completeness_check
