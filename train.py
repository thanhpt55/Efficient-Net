import os
import sys

# from imgaug import augmenters as iaa

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_folder.modules.efficientnet_wrapper import EfficientNetWrapper

CHECKPOINT_PATH = 'logs'


def main(dataset_dir, input_size, pretrained_model=None, augmentation=None):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model = EfficientNetWrapper(input_size)
    if pretrained_model:
        model.load_weight(pretrained_model, 29)
    # print('Max image size: %s' % model.check_max_image_size(dataset_dir))
    model.prepare_data(dataset_dir, batch_size=1, augmentation=augmentation)
    model.train(CHECKPOINT_PATH, gpu_count=1, architecture='B4')


if __name__ == '__main__':
    dataset_dir = 'dataset'

    input_size = 64
    #Training from scratch
    #main(dataset_dir, input_size, augmentation=None)

    #Pretraining
    pretrained_model = "E:\\Dann\\Update data\\Update\\efficientnet\\model\\"
    main(dataset_dir, input_size, pretrained_model, augmentation=None)