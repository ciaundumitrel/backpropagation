import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Model:

    def __init__(self, hidden_layers_count=None, file=None):
        self.rows_count = None
        self.cols_count = None
        self.file = file
        self.data_size = None
        self.training_data = None
        self.testing_data = None
        self.hidden_layers_size = hidden_layers_count

    def __enter__(self):
        self.file_handler = open(self.file, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            print(f"{exc_type=}")
            print(f"{exc_val=}")
            print(f"{exc_tb=}")

    def process(self):
        data = np.array([['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Y']])  # Initialize data as None

        try:
            data = pd.read_csv(self.file, sep='\t', header=None, dtype=float)
        except pd.errors.ParserError as e:
            for row in pd.read_csv(self.file, header=None)[0]:
                row_values = [float(value.strip()) for value in row.split('\t') if value]

                if data is None:
                    data = np.array([row_values])
                else:
                    data = np.concatenate([data, np.array([row_values])])

        self.rows_count = data.shape[0]
        self.cols_count = data.shape[1]

        df = pd.DataFrame(data[1:], columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Y'])
        self.training_data, self.testing_data = train_test_split(df, test_size=0.2, random_state=42)

        print(self.data_size)
