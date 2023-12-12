import keras
import tensorflow as tf
from keras.applications import MobileNetV2 as keras_model
from keras import models, layers
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from keras.optimizers import Adam
from metrics_utils import dice_loss, dice_coef, iou

class GetMobileNetV2Classifier:
    def __init__(self, input_shape: tuple,
                 weights):
        self.input_shape = self._solve_input_shape(input_shape)
        self.weigths = weights
        self.model: keras.models.Model = keras.models.Model()
        self.base_pretrained_model = None

        self._create_pretrained_model()

    def _solve_input_shape(self, input_shape):
        """
        Transform tuple (h, w) -> (h, w, 3) if necessary.
        :param input_shape: tuple with image size
        :return: corrected input shape for using in keras.Model: (h, w, c)
        """
        assert len(input_shape) in [2,3], "Invalid Input Shape Size"  # input_shape should be (h,w,c) or (h,w)
        if len(input_shape) == 3:
            return input_shape
        elif len(input_shape) == 2:
            return input_shape + (3,)

    def _create_pretrained_model(self) -> None:
        """
        create base model with pretrained weights and freeze layers
        Basically, it is MobileNetV2 (.75) with imagenet weights
        """
        self.base_pretrained_model = keras_model(input_shape=self.input_shape,
                                                 weights=self.weigths,
                                                 include_top=False,
                                                 alpha=0.75)
        self.base_pretrained_model.trainable = False

    def unfreeze_base_model(self, unfreeze_from_layer: int = 93) -> None:
        """
        :param unfreeze_from_layer: index of layer from which start training.
        209 ~ 're_lu_29' last trainable layer of MobileNetV3Small
        93 ~ 'block_10_expand_relu' last trainable layer for MobileNetV2
        """
        self.base_pretrained_model.trainable = True

        for layer in self.base_pretrained_model.layers[:unfreeze_from_layer]:
            layer.trainable = False

    def recompile_model(self,
                    loss='binary_crossentropy',
                    metrics=None,
                    optimizer=Adam,
                    learning_rate=1e-4,
                    summary=True):
        """
        Recompile new model, if something changed.
        This method is used after unfreezing certain layers of base model
        """

        if metrics is None:
            metrics = ['binary_accuracy', 'AUC']

        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics)

        if summary:
            self.model.summary()

    def build_model(self,
                    model_name="is_ship_classifier",
                    gaussian_noise=0.1,
                    dropout=0.5,
                    dense_count=128,
                    loss='binary_crossentropy',
                    metrics=None,
                    optimizer=Adam,
                    learning_rate=1e-4,
                    summary=True):

        """
        Create model based on base pretrained model. Add some input and output layers for Classification task
        """

        if metrics is None:
            metrics = ['binary_accuracy', 'AUC']

        img_in = layers.Input(self.input_shape, name='Image_RGB_In')
        img_noise = layers.GaussianNoise(gaussian_noise)(img_in)
        pt_features = self.base_pretrained_model(img_noise)

        bn_features = layers.BatchNormalization()(pt_features)
        feature_dropout = layers.SpatialDropout2D(dropout)(bn_features)
        gmp_dr = layers.GlobalMaxPooling2D()(feature_dropout)
        dr_steps = layers.Dropout(dropout)(layers.Dense(dense_count, activation='relu')(gmp_dr))
        out_layer = layers.Dense(1, activation='sigmoid')(dr_steps)

        self.model = models.Model(inputs=[img_in], outputs=[out_layer], name=model_name)

        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics)

        if summary:
            self.model.summary()


class GetMobileUnetSegmentation:
    def __init__(self,
                 input_shape: tuple = (224, 224),
                 weights = None):

        self.weights = weights
        self.input_shape = self._solve_input_shape(input_shape)

    def conv_block(self, inputs: tf.Tensor, num_filters: int) -> tf.Tensor:
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, inputs: tf.Tensor, skip: tf.Tensor, num_filters: int) -> tf.Tensor:
        "simple U-NET decoder block (conv + skip connection)"
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip])
        x = self.conv_block(x, num_filters)

        return x

    # TODO: move to utils
    def _solve_input_shape(self, input_shape):
        """
        Transform tuple (h, w) -> (h, w, 3) if necessary.
        :param input_shape: tuple with image size
        :return: corrected input shape for using in keras.Model: (h, w, c)
        """
        assert len(input_shape) in [2,3], "Invalid Input Shape Size"  # input_shape should be (h,w,c) or (h,w)
        if len(input_shape) == 3:
            return input_shape
        elif len(input_shape) == 2:
            return input_shape + (3,)

    def build_mobilenetv2_unet(self,
                            model_name = "MobileNetV2_U-Net",
                            loss = dice_loss,
                            metrics = None,
                            optimizer = Adam,
                            learning_rate = 1e-4,
                            summary = True):

        """
        Create MobileNetv2 - U-NET model, based on pretrained MobileNetV2
        """

        if metrics is None:
            metrics = [dice_coef, iou]

        """ Input """
        inputs = Input(shape=self.input_shape)

        # """ Pre-trained MobileNetV2 """
        encoder = keras_model(include_top=False, weights=self.weights,
            input_tensor=inputs, alpha=0.75)

        """ Encoder """
        s1 = encoder.get_layer("input_1").output  ## (224 x 224)
        s2 = encoder.get_layer("block_1_expand_relu").output  ## (112, 112, 96)
        s3 = encoder.get_layer("block_3_expand_relu").output  ## (56, 56, 144)
        s4 = encoder.get_layer("block_6_expand_relu").output  ## (28, 28, 144)

        """ Bridge """
        b1 = encoder.get_layer("block_13_expand_relu").output  ## (14, 14, 432)

        """ Decoder """
        d1 = self.decoder_block(b1, s4, 224)  ## (28, 28, 224)
        print(d1)
        d2 = self.decoder_block(d1, s3, 112)  ## (56, 56, 112)
        print(d2)
        d3 = self.decoder_block(d2, s2, 56)  ## (112, 112, 56)
        print(d3)
        d4 = self.decoder_block(d3, s1, 28)  ## (224, 224, 28)
        print(d4)

        """ Output """
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        self.model = keras.models.Model(inputs, outputs, name=model_name)

        self.model.compile(optimizer=optimizer(lr=learning_rate),
                           loss=loss,
                           metrics=metrics)

        if summary:
            self.model.summary()

    def compile_only_decoder(self,
                            loss = dice_loss,
                            metrics = None,
                            optimizer = Adam,
                            learning_rate = 5e-3,
                            summary = True):

        """
        Freeze encoder part and compile model with only decoder trainable
        """

        if metrics is None:
            metrics = [dice_coef, iou]

        self.model.trainable = True

        for layer in self.model.layers[:119]:
            layer.trainable = False

        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics)

        if summary:
            self.model.summary()

    def unfreeze_last_encoder_layers(self,
                            loss = dice_loss,
                            metrics = None,
                            optimizer = Adam,
                            learning_rate = 5e-4,
                            summary = True):

        """
        Unfreeze certain layers in decoder and compile model with trainable encoder and patly decoder
        """

        if metrics is None:
            metrics = [dice_coef, iou]

        self.model.trainable = True

        for layer in self.model.layers[:93]:
            layer.trainable = False

        self.model.compile(optimizer=optimizer(lr=learning_rate),
                           loss=loss,
                           metrics=metrics)

        if summary:
            self.model.summary()