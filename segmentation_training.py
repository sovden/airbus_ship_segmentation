from models import GetMobileUnetSegmentation
from data_prep import GetSegmentationDataGenerator

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
NUM_EPOCHS_DECODER = 1
NUM_EPOCHS_ALL = 2
MODELS_DIR = "saved_models/"
WEIGHTS = f"{MODELS_DIR}just_mobilenet_224_075.hdf5" # "imagenet"

if __name__ == "__main__":

    segmentation_generator = GetSegmentationDataGenerator(batch_size=BATCH_SIZE, target_size=IMG_SIZE)
    aug_gen = segmentation_generator.get_train_generator()
    val_gen = segmentation_generator.get_valid_generator()

    train_size = segmentation_generator.data_size_dict["train"]
    valid_size = segmentation_generator.data_size_dict["valid"]

    print(f"train_size: {train_size}, valid_size: {valid_size}")

    segmentation = GetMobileUnetSegmentation(weights=WEIGHTS, input_shape=IMG_SIZE)
    segmentation.build_mobilenetv2_unet(summary=False)
    segmentation.compile_only_decoder(summary=False, learning_rate=1e-3)
    # print("aug_gen.n:", train_df.shape, train_df.shape[0]//8)
    segmentation.model.fit(aug_gen,
                        steps_per_epoch=train_size//BATCH_SIZE,
                        batch_size=BATCH_SIZE,
                        validation_data=val_gen,
                        validation_steps=valid_size//BATCH_SIZE//10,
                        shuffle=True,
                        epochs=NUM_EPOCHS_DECODER,
                        workers=1 # the generator is not very thread safe
                                           )

    segmentation.unfreeze_last_encoder_layers(summary=False, learning_rate=7e-4)

    segmentation.model.fit(aug_gen,
                        steps_per_epoch=train_size//BATCH_SIZE,
                        batch_size=BATCH_SIZE,
                        validation_data=val_gen,
                        validation_steps=valid_size//BATCH_SIZE//10,
                        shuffle=True,
                        epochs=NUM_EPOCHS_ALL,
                        workers=1 # the generator is not very thread safe
                                           )

    segmentation.model.save(f"{MODELS_DIR}mobilenetv2_unet.h5")
