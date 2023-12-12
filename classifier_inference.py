from keras.models import load_model
from data_prep import GetClassifierDataGenerator
from metrics_utils import f1_m
import numpy as np
from inference_utils import show_images_classifier, img_by_path, show_one_image_classifier

TARGET_SIZE = (224, 224)
BATCH_SIZE = 8
EVALUATE = False
INFERENCE_BATCH = False
INFERENCE_IMG = True
IMG_PATH = "test_image.jpg"


model = load_model('saved_models/mobilenetv2_224_075_model_version_2.h5')

if INFERENCE_IMG:
    img_array = img_by_path(IMG_PATH, TARGET_SIZE)
    yp = np.array(model.predict(np.expand_dims(img_array, 0)))
    yp = np.where(yp > 0.5, 1, 0).flatten()

    show_one_image_classifier(img_array, yp[0])

if EVALUATE or INFERENCE_BATCH:
    generators = GetClassifierDataGenerator(batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    test_gen = generators.get_test_generator()

    if EVALUATE:
        model.compile(metrics=[f1_m])
        scores=model.evaluate(test_gen, verbose=1)
        print(scores)

    if INFERENCE_BATCH:
        x, y = next(iter(test_gen))
        yp = np.array(model.predict(x))
        yp = np.where(yp > 0.5, 1, 0).flatten()

        show_images_classifier(x, y, yp, batch_size=BATCH_SIZE)
