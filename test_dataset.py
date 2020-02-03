import os
import sys
import numpy as np
import cv2

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_folder.modules.efficientnet_wrapper import EfficientNetWrapper



def main(model_dir, epoch, test_dir, thresholds):
    model = EfficientNetWrapper(64)
    model.load_weight(model_dir, epoch)
    cf_matrix = [[0, 0], [0, 0]]
    counter = [0, 0]
    casename_matrix = [["TN", "FN"], ["FP", "TP"]]
    with open('%s/labels.txt' % test_dir) as f:
        while True:
            img_name = f.readline().strip()
            if not img_name:
                break
            print(img_name)
            class_name = int(f.readline().strip())
            img = cv2.imread('%s/%s' % (test_dir, img_name))
            img_tosave = img.copy()
            prediction = model.predict_one(img, thresholds)
            # onehotPred = np.zeros((2))
            # onehotPred[prediction[0]] = 1
            counter[class_name] += 1
            cf_matrix[class_name][prediction[0]] += 1
            casename = casename_matrix[prediction[0]][class_name]
            print(cv2.imwrite(os.path.join("E:/Dann/Update data/Update/efficientnet/dataset/Source/TEST_SQ6101/test/" + casename, img_name), img_tosave))



    print(cf_matrix)
    print(counter)
            # print('GT: %s' % class_name)
            # print('Pred: %s - %s' % (prediction[2], prediction[1]))
            # cv2.imshow(img_name, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # img_name = f.readline().strip()
        # class_name = f.readline().strip()
        # img = cv2.imread('%s/%s' % (test_dir, img_name))
        # prediction = model.predict_one(img, thresholds)
        # # onehotPred = np.zeros((2))
        # # onehotPred[]


if __name__ == '__main__':
    model_dir = "E:\\Dann\\Update data\\Update\\efficientnet\\logs\\20200131165323\\"
    epoch = 22
    test_dir = 'E:\\Dann\\Update data\\Update\\efficientnet\\dataset\\Source\\TEST_SQ6101\\test\\'
    thresholds = None
    main(model_dir, epoch, test_dir, thresholds)
