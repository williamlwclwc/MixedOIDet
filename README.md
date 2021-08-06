# Mixed Open Images Detector For Nocaps

## Referenced Detectors

* We combined the following detectors together to achieve better detection results for all 600 classes of Open Images without retraining: 
    * [Nocaps Provided Detector](https://github.com/nocaps-org/image-feature-extractors)
    * [2020 Open Images Challenge #1 - UniDet](https://github.com/xingyizhou/UniDet)

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
* `--output_format`: Format to save the combined detection (currently only tsv is supported, do not need to specify this for now)

### Examples

* Nocaps data inference: 
    ```shell
    python run.py --input_imgs /data/private/liuwenchang/scene_graph_benchmark/others/nocaps_data/images_val \
    --input_annotations /data/private/liuwenchang/scene_graph_benchmark/others/nocaps_data/nocaps_val_image_info.json \
    --output ./output/nocaps_val.tsv
    ```
* COCO data inference:
    ```shell
    python run.py --input_imgs /data/private/liuwenchang/UniDet/datasets/coco/train2014 \
    --input_annotations /data/private/liuwenchang/UniDet/datasets/coco/annotations/instances_train2014.json \
    --output ./output/coco_train.tsv
    ```

