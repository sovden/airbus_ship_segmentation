from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd
import PIL
from utils.data_utils import masks_as_image, resize_mask_array

TRAIN_V2_DIR_PATH = "C:/Users/Denys/Documents/ml/winstar_internship/train_v2"

class GetClassifierDataGenerator:
    def __init__(self, batch_size: int = 8,
                 target_size: tuple = (244, 244),
                 csv_dir: str = "dataframes",
                 data_path: str = TRAIN_V2_DIR_PATH):
        self.batch_size = batch_size
        self.target_size = target_size
        self.csv_dir = csv_dir
        self.data_path = data_path

        self._get_train_valid_test_df()
        self._get_image_generators_args()

    def _get_train_valid_test_df(self):
        """
        declare train/valid/test data from prepared .csv
        """

        # train dataset balanced (0 / 1 ~ 50%/50%)
        dataframe = pd.read_csv(os.path.join(self.csv_dir, "train_ship_segmentations.csv"))
        self.train_df = self.dataframe_preprocess(dataframe, balanced=True)

        dataframe = pd.read_csv(os.path.join(self.csv_dir, "valid_ship_segmentations.csv"))
        self.valid_df = self.dataframe_preprocess(dataframe)

        dataframe = pd.read_csv(os.path.join(self.csv_dir, "test_ship_segmentations.csv"))
        self.test_df = self.dataframe_preprocess(dataframe)

    def dataframe_preprocess(self, dataframe, balanced=False) -> pd.DataFrame:
        """
        Simple preprocessing of .csv data for utils task
        """
        dataframe = dataframe[["ImageId", "ships", "has_ship"]].drop_duplicates()
        dataframe["has_ship_vec"] = dataframe['has_ship'].map(lambda x: [x])
        dataframe["path"] = dataframe["ImageId"].apply(lambda x: os.path.join(self.data_path, x))

        ship_dataframe = dataframe.loc[dataframe["has_ship"] == 1]
        non_ship_dataframe = dataframe.loc[dataframe["has_ship"] == 0]

        if balanced:
            # SIMPLE UNDERSAMPLING OF IMAGES WITHOUT SHIPS
            non_ship_dataframe = non_ship_dataframe.sample(ship_dataframe.shape[0])
            dataframe = pd.concat([ship_dataframe, non_ship_dataframe], ignore_index=True)

        print(f"class balance, 1: {ship_dataframe.shape[0]}, 0: {non_ship_dataframe.shape[0]}")

        return dataframe

    def _get_image_generators_args(self):
        self.dg_args = dict(
            featurewise_center = False,
                          samplewise_center = False,
                          rotation_range = 45,
                          width_shift_range = 0.1,
                          height_shift_range = 0.1,
                          shear_range = 0.01,
                          zoom_range = [0.9, 1.25],
                          brightness_range = [0.5, 1.5],
                          horizontal_flip = True,
                          vertical_flip = True,
                          fill_mode = 'reflect',
                            data_format = 'channels_last')

        self.valid_args = dict(fill_mode = 'reflect',
                           data_format = 'channels_last')

        self.test_args = dict(fill_mode = 'reflect',
                           data_format = 'channels_last')

    def flow_from_dataframe(self, img_data_gen, in_df, path_col, y_col, **dflow_args) -> ImageDataGenerator:
        """
        Create custom DataGenerator based on the ImageDataGenerator with/without augumentation
        """
        base_dir = os.path.dirname(in_df[path_col].values[0])
        print('## Ignore next message from keras, values are replaced anyways')
        df_gen = img_data_gen.flow_from_directory(base_dir,
                                         class_mode = 'sparse',
                                        **dflow_args)
        df_gen.filenames = in_df[path_col].values
        df_gen.filepaths.extend(df_gen.filenames)
        df_gen.classes = np.stack(in_df[y_col].values)
        df_gen.samples = in_df.shape[0]
        df_gen.n = in_df.shape[0]
        df_gen._set_index_array()
        df_gen.directory = '' # since we have the full path
        print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
        return df_gen

    def get_train_generator(self):
        core_idg = ImageDataGenerator(**self.dg_args)
        train_gen = self.flow_from_dataframe(core_idg, self.train_df,
                                        path_col='path',
                                        y_col='has_ship_vec',
                                        target_size=self.target_size,
                                        color_mode='rgb',
                                        batch_size=self.batch_size,
                                        shuffle=True)

        return train_gen

    def get_valid_generator(self):
        valid_idg = ImageDataGenerator(**self.valid_args)
        valid_gen = self.flow_from_dataframe(valid_idg, self.valid_df,
                                             path_col='path',
                                             y_col='has_ship_vec',
                                             target_size=self.target_size,
                                             color_mode='rgb',
                                             batch_size=self.batch_size,
                                             shuffle=True)

        return valid_gen

    def get_test_generator(self):
        test_idg = ImageDataGenerator(**self.test_args)
        test_gen = self.flow_from_dataframe(test_idg, self.test_df,
                                             path_col='path',
                                             y_col='has_ship_vec',
                                             target_size=self.target_size,
                                             color_mode='rgb',
                                             batch_size=self.batch_size,
                                            shuffle=False)

        return test_gen


class GetSegmentationDataGenerator:
    def __init__(self, batch_size: int = 8,
                 image_scaling: tuple = (3, 3),
                 csv_dir: str = "dataframes",
                 data_path: str = TRAIN_V2_DIR_PATH,
                 target_size: tuple = (224, 224)):
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_scaling = image_scaling
        self.csv_dir = csv_dir
        self.data_path = data_path
        self.data_size_dict = {"train": 0, "valid": 0, "test": 0}

        self._get_train_valid_test_df()
        self._define_image_generators_args()

    def _get_train_valid_test_df(self):
        """
        declare train/valid/test data from prepared .csv
        """

        # train dataset balanced. Images without ships ~ 30% of all and equals number of images with 1 or 2 ships
        # this is one of the many approaches, that could be tested
        dataframe = pd.read_csv(os.path.join(self.csv_dir, "train_ship_segmentations.csv"))
        self.train_df = self.dataframe_preprocess(dataframe, balanced=True, df_name="train_df")
        self.data_size_dict["train"] = self.train_df[["ImageId"]].drop_duplicates().shape[0]

        dataframe = pd.read_csv(os.path.join(self.csv_dir, "valid_ship_segmentations.csv"))
        self.valid_df = self.dataframe_preprocess(dataframe, df_name="valid_df")
        self.data_size_dict["valid"] = self.valid_df[["ImageId"]].drop_duplicates().shape[0]

        dataframe = pd.read_csv(os.path.join(self.csv_dir, "test_ship_segmentations.csv"))
        self.test_df = self.dataframe_preprocess(dataframe, df_name="test_df")
        self.data_size_dict["test"] = self.test_df[["ImageId"]].drop_duplicates().shape[0]

    def dataframe_preprocess(self, dataframe, balanced=False, df_name = "dataframe"):
        dataframe["path"] = dataframe["ImageId"].apply(lambda x: os.path.join(self.data_path, x))

        ship_dataframe = dataframe.loc[dataframe["has_ship"] == 1]
        non_ship_dataframe = dataframe.loc[dataframe["has_ship"] == 0]

        if balanced:
            # SIMPLE UNDERSAMPLING OF IMAGES WITHOUT SHIPS
            sample_for_zero = dataframe.loc[dataframe["ships"].isin([1, 2])].shape[0]
            non_ship_dataframe = non_ship_dataframe.sample(sample_for_zero)
            dataframe = pd.concat([ship_dataframe, non_ship_dataframe], ignore_index=True)

        print(f"{df_name} class balance, 1: {ship_dataframe[['ImageId']].shape[0]}, 0: {non_ship_dataframe[['ImageId']].shape[0]}")

        return dataframe

    def _define_image_generators_args(self):
        self.dg_args = dict(
            featurewise_center=False,
            samplewise_center=False,
            rotation_range=45,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,
            zoom_range=[0.9, 1.25],
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            data_format='channels_last')

        self.valid_args = dict(fill_mode = 'reflect',
                           data_format = 'channels_last')

        self.test_args = dict(fill_mode = 'reflect',
                           data_format = 'channels_last')

    def make_image_gen(self, dataframe):
        """
        Python generator that is inner part of DataGenerator for trainig.
        It solves issue of creating mask for each img for each batch.
        It takes data by the path from dataframe.
        Data are raw image (c_img) and segmentation mask (c_mask_array).
        If we work with low res img, resizing applied
        """
        # TODO: make it more robust, especially in terms of threading
        all_batches = list(dataframe.groupby('ImageId'))
        out_rgb = []
        out_mask = []
        while True:
            np.random.shuffle(all_batches)
            for c_img_id, c_masks in all_batches:
                # TODO: we already have this data in .csv data
                rgb_path = os.path.join(self.data_path, c_img_id)
                c_img = PIL.Image.open(rgb_path)
                c_mask_array = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
                if self.target_size != (768, 768):
                    c_img = c_img.resize(self.target_size).convert("RGB")
                    c_mask_array = resize_mask_array(c_mask_array, self.target_size)
                c_img_array = np.array(c_img)
                out_rgb += [c_img_array]
                out_mask += [c_mask_array]
                if len(out_rgb)>=self.batch_size:
                    yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                    out_rgb, out_mask= [], []

    # TODO: make thread stable generator
    def create_aug_gen(self, in_gen, idg_args, seed=None):
        """
        Custom datagenerator based on  ImageDataGenerator
        and make_image_gen() custom python gen
        """
        image_gen = ImageDataGenerator(**idg_args)
        label_gen = ImageDataGenerator(**idg_args)
        np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
        for in_x, in_y in in_gen:
            seed = np.random.choice(range(9999))
            # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
            g_x = image_gen.flow(255 * in_x,
                                 batch_size=in_x.shape[0],
                                 seed=seed,
                                 shuffle=True)
            g_y = label_gen.flow(in_y,
                                 batch_size=in_x.shape[0],
                                 seed=seed,
                                 shuffle=True)

            yield next(g_x) / 255.0, next(g_y)

    def get_train_generator(self):
        image_generator = self.make_image_gen(self.train_df[["ImageId", "EncodedPixels"]])
        aug_gen = self.create_aug_gen(image_generator, self.dg_args)

        return aug_gen

    def get_valid_generator(self):
        image_generator = self.make_image_gen(self.valid_df[["ImageId", "EncodedPixels"]])
        aug_gen = self.create_aug_gen(image_generator, self.dg_args)

        return aug_gen

    def get_test_generator(self):
        image_generator = self.make_image_gen(self.test_df[["ImageId", "EncodedPixels"]])
        aug_gen = self.create_aug_gen(image_generator, self.dg_args)

        return aug_gen