
from model import Model

if __name__ == '__main__':
    file_name = 'seeds/seeds_dataset.txt'

    model = Model(file=file_name, hidden_layers_count=3)
    model.process_data()

    model.build_neural_network()

    model.extract_features_and_targets()
    model.train_neural_network()

    input_example = [0.5, 0.2, 0.8, 0.3, 0.1, 0.7, 0.4]
    target_example = 0.6

    output, _ = model.forward_propagation(input_example)

    model.backward_propagation(input_example, target_example)
