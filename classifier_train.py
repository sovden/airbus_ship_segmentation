import tensorflow as tf
from keras import backend as K
print("gpu:", tf.config.list_physical_devices('GPU'))
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
K.set_session(sess)

from data_prep import GetClassifierDataGenerator
from models import GetMobileNetV2Classifier
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

IMG_SIZE = (224, 224) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 16 # [1, 8, 16, 24]
WEIGHTS = 'imagenet' # None
NUM_EPOCHS_HEAD = 1
NUM_EPOCHS_FILTERS = 4
MODELS_DIR = "saved_models/"

SHOW_MODELS = False

if __name__ == "__main__":

    if SHOW_MODELS:
        classifier = GetMobileNetV2Classifier(input_shape=(224, 224, 3), weights=WEIGHTS)
        classifier.base_pretrained_model.summary()
        classifier.build_model()
        # classifier.model.summary()
        classifier.unfreeze_base_model()
        classifier.recompile_model()
        # classifier.model.summary()

        just_mobilenet = classifier.model.get_layer("mobilenetv2_0.75_224")
        just_mobilenet.summary()

    else:
        generators = GetClassifierDataGenerator(batch_size=BATCH_SIZE, target_size=IMG_SIZE)
        train_gen = generators.get_train_generator()
        valid_gen = generators.get_valid_generator()

        # CHECK BATCH IMG SHAPE
        t_x, t_y = next(train_gen)
        print("t_x.shape[1:]", t_x.shape)
        classifier = GetMobileNetV2Classifier(input_shape=t_x.shape[1:],
                                              weights=WEIGHTS)

        classifier.build_model(learning_rate=2e-3,
                               model_name="ship_classifier",
                               summary=False)

        def sheduler_function(epoch, lr):
            if epoch <= 2:
                return lr
            else:
                return lr*0.5

        scheduler = LearningRateScheduler(sheduler_function)
        callbacks_list = [scheduler]

        print("START TRAINING OF HEAD")
        classifier.model.fit(
            train_gen,
            steps_per_epoch=train_gen.n//BATCH_SIZE,
            validation_data=valid_gen,
            validation_steps=valid_gen.n//BATCH_SIZE//10,
            epochs=NUM_EPOCHS_HEAD,
            callbacks=callbacks_list,
            shuffle=True,
            workers=4)

        classifier.unfreeze_base_model()
        classifier.recompile_model(learning_rate=2e-3, summary=False)

        print("START TRAINING OF LAST LAYERS")
        classifier.model.fit(
            train_gen,
            steps_per_epoch=train_gen.n//BATCH_SIZE,
            validation_data=valid_gen,
            validation_steps=valid_gen.n//BATCH_SIZE//10,
            epochs=NUM_EPOCHS_FILTERS,
            callbacks=callbacks_list,
            shuffle=True,
            workers=4)

        # keras.Sequential classifier.model.layers

        classifier.model.save(f"{MODELS_DIR}mobilenetv2_224_075.h5")

        just_mobilenet = classifier.model.get_layer(f"mobilenetv2_0.75_224")
        just_mobilenet.save(f"{MODELS_DIR}just_mobilenet_224_075.h5")
        just_mobilenet.save_weights(f"{MODELS_DIR}just_mobilenet_224_075_weights.hdf5")