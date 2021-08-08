# Mixed Open Images Detector For Nocaps

* This repo contains an automatic script that combines the results of Pretrained 
[Nocaps object detector](https://github.com/nocaps-org/image-feature-extractors) and [UniDet detector](https://github.com/xingyizhou/UniDet) 
to produce better predictions for all 600 classes of Open Images / Nocaps 

## Referenced Detectors

* We combined the following detectors together to achieve better detection results for all 600 classes of Open Images without retraining: 
    * [Nocaps Provided Detector](https://github.com/nocaps-org/image-feature-extractors)
    * [2020 Open Images Challenge #1 - UniDet](https://github.com/xingyizhou/UniDet)

## How we combine the results

* We select one of the detectors' predictions for each class based on the mAP score experiment(see `./data/sub_list.json`, classes in that file we will use Nocaps detector, you can use a different json file), however, for some images for example, mAP score for class A of detector-1 is higher than of detector-2, but in this specific image, detector-1 makes no predictions while detector-2 predicts class A objects, in this case we will use detector-2 for that image, otherwise many images will have no bboxes output.

* The list of images with no output predictions:
    * Nocaps validation: all images have predictions
    * Nocaps test: see `./output/nocaps_test_nooutput.txt`
    * COCO train: see `./output/coco_train_nooutput.txt`

## Installation

* Install Nocaps Image-Feature-extractors
    * Clone repo from [Nocaps Provided Detector](https://github.com/nocaps-org/image-feature-extractors) and put it in the same dir as MixedOIDet
    * `OI Detector`: download from [here](https://www.dropbox.com/s/uoai4xqfdx96q2c/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018.tar.gz) and un-tar it in `models` sub-directory.
    * Build OI detector docker file: 
      ```shell
        docker build --file Dockerfile_oi --tag oi_image .
      ```
    * More Details can be found in the [Nocaps repo](https://github.com/nocaps-org/image-feature-extractors)
    
* Install UniDet
    * Clone repo from [UniDet](https://github.com/xingyizhou/UniDet) and put it in the same dir as MixedOIDet
    * Replace the `UniDet/projects/UniDet/demo/demo.py` with the `demo.py` in MixedOIDet
    * Install UniDet as described [here](https://github.com/xingyizhou/UniDet/blob/master/INSTALL.md)
    * Download the [model](https://drive.google.com/file/d/1PZ_EQDfCSkmiaJobrCRddu6Bf6QdU1LB/view?usp=sharing) and put it into `UniDet/models/`
    * Other models and more details can be found [here](https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet_docs/REPRODUCE.md)

## How to Run

### Arguments

* `--input_imgs`: The path of a directory of images to be detected
* `--input_annotations`: The path of the annotation file of images
* `--output`: A file path to save output predictions
* `--nocaps_det_dir`: The path of nocaps detector image-feature-extractors (default: `../image-feature-extractors`)
* `--UniDet_dir`: Path of UniDet dir (default: `../UniDet`)
* `--UniDet_model`: Specify which UniDet model you want to use (default: `Unified_learned_OCI_R50_8x`)
* `--device`: Specify which GPU you will use for the docker of nocaps detector (default: 0)
* `--steps`: Specify which step you want to execute: 0: all 3 steps, 1: nocaps detector step only, 2: UniDet step only, 3: combination step only (default: 0)
* `--output_format`: Format to save the combined detection (currently only tsv is supported, do not need to specify this for now)
    * The tsv format we use here is the same as [Microsoft Oscar](https://github.com/microsoft/Oscar)

### Examples

* Nocaps validation data inference: 
    ```shell
    python run.py --input_imgs /data/private/liuwenchang/scene_graph_benchmark/others/nocaps_data/images_val \
    --input_annotations /data/private/liuwenchang/scene_graph_benchmark/others/nocaps_data/nocaps_val_image_info.json \
    --device 7 --output ./output/nocaps_val
    ```

* Nocaps test data inference: 
    ```shell
    python run.py --input_imgs /data/private/liuwenchang/scene_graph_benchmark/others/nocaps_data/images_test \
    --input_annotations /data/private/liuwenchang/scene_graph_benchmark/others/nocaps_data/nocaps_test_image_info.json \
    --device 7 --output ./output/nocaps_test
    ```

* COCO data inference:
    ```shell
    python run.py --input_imgs ../UniDet/datasets/coco/train2014 \
    --input_annotations /data/private/liuwenchang/UniDet/datasets/coco/annotations/instances_train2014.json \
    --device 7 --output ./output/coco_train
    ```
