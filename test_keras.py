MODEL = "MOBNET" # EFFNET
WEIGHTS = 'imagenet' # None

print(MODEL, WEIGHTS)

if MODEL == "MOBNET":
    from keras.applications import MobileNetV3Small as keras_model
    from keras.applications.mobilenet_v3 import preprocess_input
elif MODEL == "EFFNET":
    from keras.applications import EfficientNetB0 as keras_model

from data_prep import GetClassifierDataGenerator

GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# number of validation images to use
VALID_IMG_COUNT = 1000
# maximum number of training images
MAX_TRAIN_IMAGES = 100
IMG_SIZE = (299, 299) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 8 # [1, 8, 16, 24]
DROPOUT = 0.5
DENSE_COUNT = 128
LEARN_RATE = 1e-4
RGB_FLIP = 1 # should rgb be flipped when rendering images

generators = GetClassifierDataGenerator(batch_size=BATCH_SIZE, target_size=IMG_SIZE)
train_gen = generators.get_train_generator()
valid_gen = generators.get_valid_generator()

t_x, t_y = next(train_gen)
print("t_x.shape[1:]", t_x.shape[1:])



base_pretrained_model = keras_model(input_shape = t_x.shape[1:], weights=WEIGHTS, include_top=False)
base_pretrained_model.trainable = False

from keras import models, layers
from keras.optimizers import Adam

img_in = layers.Input(t_x.shape[1:], name='Image_RGB_In')
img_noise = layers.GaussianNoise(GAUSSIAN_NOISE)(img_in)
pt_features = base_pretrained_model(img_noise)

# TODO: why?
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]

bn_features = layers.BatchNormalization()(pt_features)
feature_dropout = layers.SpatialDropout2D(DROPOUT)(bn_features)
gmp_dr = layers.GlobalMaxPooling2D()(feature_dropout)
dr_steps = layers.Dropout(DROPOUT)(layers.Dense(DENSE_COUNT, activation = 'relu')(gmp_dr))
out_layer = layers.Dense(1, activation = 'sigmoid')(dr_steps)

ship_model = models.Model(inputs = [img_in], outputs = [out_layer], name = 'full_model')

ship_model.compile(optimizer = Adam(lr=LEARN_RATE),
                   loss = 'binary_crossentropy',
                   metrics = ['binary_accuracy'])

ship_model.summary()

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('boat_detector')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

train_gen.batch_size = BATCH_SIZE
ship_model.fit_generator(train_gen,
                         steps_per_epoch=train_gen.n//BATCH_SIZE,
                      # validation_data=(valid_x, valid_y),
                      epochs=5,
                      callbacks=callbacks_list,
                      workers=1)