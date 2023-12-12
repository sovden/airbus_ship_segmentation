from sklearn.model_selection import train_test_split
import PIL
import os
import numpy as np
import pandas as pd

def prepare_train_valid_test_split(dataframe_dir: str="dataframes", valid_size: float=0.07, test_size: float=0.03) -> None:
    """
    Since our model = finetuned encoder + trained decoder, for both training we need the same
    split of data.

    :param dataframe_dir: path to .csv files
    :param test_size:
    :return: None (just save related dataframes)
    """
    raw_df = pd.read_csv(os.path.join(dataframe_dir, 'train_ship_segmentations_v2.csv'))
    raw_df['path'] = raw_df['ImageId'].map(lambda x: os.path.join(dataframe_dir, x))

    raw_df['ships'] = raw_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = raw_df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
    train_ids, valid_ids = train_test_split(unique_img_ids,
                                            test_size=valid_size+test_size,
                                            stratify=unique_img_ids['ships'])

    valid_ids, test_ids = train_test_split(valid_ids,
                                            test_size=test_size/(valid_size+test_size),
                                            stratify=valid_ids['ships'])

    train_df = pd.merge(raw_df, train_ids)
    print(f"train_df size: {train_df.shape[0]}, unique images: {train_df['ImageId'].drop_duplicates().shape[0]}")
    train_df.to_csv(os.path.join(dataframe_dir, "train_ship_segmentations.csv"), index=False)

    valid_df = pd.merge(raw_df, valid_ids)
    print(f"valid_df size: {valid_df.shape[0]}, unique images: {valid_df['ImageId'].drop_duplicates().shape[0]}")
    valid_df.to_csv(os.path.join(dataframe_dir, "valid_ship_segmentations.csv"), index=False)

    test_df = pd.merge(raw_df, test_ids)
    print(f"valid_df size: {test_df.shape[0]}, unique images: {test_df['ImageId'].drop_duplicates().shape[0]}")
    test_df.to_csv(os.path.join(dataframe_dir, "test_ship_segmentations.csv"), index=False)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

def resize_mask_array(mask_array, target_size):
    img = PIL.Image.fromarray(mask_array[:, :, 0]).resize(target_size)
    np_array = np.expand_dims(np.array(img).astype(np.float32), -1)
    return np_array