import os

import sys


current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_folder.modules.efficientnet_wrapper import EfficientNetWrapper


def main(model_architect_path, epoch, dataset_dir):
    model = EfficientNetWrapper(150)
    model.prepare_data(dataset_dir)
    model.load_weight(model_architect_path, epoch)
    model.evaluate('test')


if __name__ == '__main__':
    model_dir = 'model'
    epoch = 456
    dataset_dir = 'dataset'
    main(model_dir, epoch, dataset_dir)
