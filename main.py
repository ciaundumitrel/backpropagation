
from model import Model

if __name__ == '__main__':
    file_name = 'seeds/seeds_dataset.txt'

    with Model(file=file_name) as model:
        model.process()
