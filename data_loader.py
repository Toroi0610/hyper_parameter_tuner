import pandas as pd

def data_loader(filepath, target_name):
    data = pd.read_csv(filepath)
