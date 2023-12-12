import tensorflow as tf
from keras import backend as K
print("gpu:", tf.config.list_physical_devices('GPU'))
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
K.set_session(sess)

from data_prep import GetClassifierDataGenerator, GetSegmentationDataGenerator
from models import GetMobileNetV2Classifier, GetMobileUnetSegmentation
from lr_finder import LRFinder
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
# WEIGHTS = 'imagenet'
MODELS_DIR = "../saved_models/"
WEIGHTS = f"{MODELS_DIR}just_mobilenet_224_075.hdf5" # "imagenet"
DATA_FRAC_TO_SEARCH = 0.3

# generators = GetClassifierDataGenerator(batch_size=BATCH_SIZE, target_size=IMG_SIZE)
# if DATA_FRAC_TO_SEARCH < 1:
#     generators.train_df = generators.train_df.sample(frac=DATA_FRAC_TO_SEARCH)
#     print("new balance:", generators.train_df["has_ship"].value_counts())
# train_gen = generators.get_train_generator()
#
# utils = GetMobileNetV2Classifier(input_shape=IMG_SIZE, weights=WEIGHTS)
# utils.build_model(summary=False, model_name="lr_finder_classifier")
# model = utils.model

segmentation_generator = GetSegmentationDataGenerator(batch_size=BATCH_SIZE)
if DATA_FRAC_TO_SEARCH < 1:
    segmentation_generator.train_df = segmentation_generator.train_df.sample(frac=DATA_FRAC_TO_SEARCH)
    print("new balance:", segmentation_generator.train_df["has_ship"].value_counts())
train_gen = segmentation_generator.get_train_generator()

segmentation = GetMobileUnetSegmentation(weights=WEIGHTS)
segmentation.build_mobilenetv2_unet(summary=False, model_name="lr_finder_segmentation")
segmentation.compile_only_decoder(summary=False)
model = segmentation.model

lr_finder = LRFinder(model)
lr_finder.find(train_gen, segmentation_generator.train_df.shape[0],
               start_lr=1e-6,
               end_lr=5,
               batch_size=BATCH_SIZE,
               epochs=5)
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
plt.show()