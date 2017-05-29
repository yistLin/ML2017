import pandas as pd


def read_data(data_path):
    datas = pd.read_csv(data_path, sep=',')
    return datas.values


def write_output(outputs, output_path):
    cols = {
        'TestDataID': range(1, len(outputs)+1),
        'Rating': outputs
        }
    df = pd.DataFrame(cols)
    df.to_csv(output_path, index=False, columns=['TestDataID', 'Rating'])
