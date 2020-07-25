import pandas as pd

def data_loader(filepath, target_list:list):
    data = pd.read_csv(filepath)
    columns = list(data.columns)
    target = data.loc[:, target_list]
    input_columns = columns
    for target_column in target_list:
        columns.remove(target_column)

    input_data = data.loc[:, input_columns]

    return input_data, target