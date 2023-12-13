# AirBus ship Segmentation MobileNetV2 based U-NET

## Data
EDA could be found in EDA.ipynb. Data is images in train_v2 dir and Dataframe dataframes/train_ship_segmentations_v2.csv with ImageId (path) and Mask for each ship
Main insights:
1. Heavy imbalanced data set for segmentation task, ~ 1 / 10k pixels ships' pixels / all pixels
2. Mainly water with a little part of land/ports
3. Masks for segmentation provided in specific format
4. Each row in dataframe is certain ship, not image with masks for all ships

## Model
It was decided to use UNET with pretrained encoder. For simplicity and time saving MobileNetV2 as encoder was choosed with imagenet weights.This model is relatively simple architecture and low number of parameters while good performance. 
For better performace, Mobilenet was pretrained with a few epoch as ship classifier. After this, encoder was built from mobilenet layers was and Unet trained.
1. Fine Tune MobileNetV2 imagenet on train dataset (balanced dataset, with undersampling of no ship images)
   - 1 epoch just head;
   - a few epochs for head + last conv layers -> ~80% bin_acc
2. Encoder(Finetune mobilenet) + custom Decoder (4 layers of convs+skip connection)
   - 1 epoch just decoder;
   - a few epochs for whole model;

## Results
For scoring dice score was used. For a few epoch training with low batch size (8 / 16) resulted model gives dice 0.75 for only ships images. For all images dice ~0.65 for all images, using of classifier to filter FP will probably improve results.

## Inference
Any segmentation_inference.py and inference.ipynb could be used for testing/inference. In case of batch testing - provide correct path to the train data in data_prep.py: _TRAIN_V2_DIR_PATH = "../train_v2"_

## Possible improvements
1. More epochs (trined for <10 epochs both classifier and unet)
2. Play more with LR, simple lr sheduling for epochs was useful
3. Try different size of network
4. Try different architectures, ResNet should be better, but it bigger 10x in comparison to the MobileNetV2
5. Code refactoring

