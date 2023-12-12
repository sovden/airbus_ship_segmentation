from keras.models import load_model
from data_prep import GetSegmentationDataGenerator
from metrics_utils import dice_loss, dice_coef, iou
from inference_utils import show_images_segmentation, img_by_path, show_one_image_segmentation, compute_dice_for_inference
from data_utils import resize_mask_array
import numpy as np

TARGET_SIZE = (224, 224)
BATCH_SIZE = 16
EVALUATE = False
INFERENCE_BATCH = True
INFERENCE_IMG = False
IMG_PATH = "test_image.jpg"


model = load_model('saved_models/mobilenetv2_unet.h5', custom_objects={"dice_loss": dice_loss,
                                                                       "dice_coef": dice_coef,
                                                                       "iou": iou})
if INFERENCE_IMG:
    img_array = img_by_path(IMG_PATH, TARGET_SIZE)
    yp = np.array(model.predict(np.expand_dims(img_array / 255.0, 0)))
    yp = np.where(yp > 0.5, 1, 0)

    show_one_image_segmentation(img_array, yp[0])

if EVALUATE or INFERENCE_BATCH:
    generators = GetSegmentationDataGenerator(batch_size=BATCH_SIZE)
    # df = generators.test_df.copy()
    # df = df.loc[df["has_ship"] == 1]
    # generators.test_df = df
    test_gen = generators.get_test_generator()

    if EVALUATE:
        scores = model.evaluate(test_gen, verbose=1)
        print(scores)

    if INFERENCE_BATCH:
        x, y = next(iter(test_gen))
        yp = np.array(model.predict(x))
        yp = np.where(yp > 0.5, 1.0, 0.0)

        coeficients = [float(compute_dice_for_inference(resize_mask_array(y[i], TARGET_SIZE), yp[i]))
                       for i in range(BATCH_SIZE)]
        print(coeficients)
        show_images_segmentation(x, y, yp,
                                 # image_paths=
                                 dice_coefs=coeficients,
                                 batch_size=BATCH_SIZE)