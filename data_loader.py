from typing import Union, List
import pandas as pd

def data_loader(filepath, target_list: Union[None, list]):
    data = pd.read_csv(filepath)
    columns = list(data.columns)
    input_columns = columns
    target = None
    if target_list is not None:
        for target_column in target_list:
            input_columns.remove(target_column)
        target = data.loc[:, target_list]

    input_data = data.loc[:, input_columns]

    return input_data, target