import os
import argparse
import json
import csv
import re


def get_parser():
    parser = argparse.ArgumentParser(description="Mixed Detector for Open Images 600 classes")
    parser.add_argument(
        "--input_imgs",
        help="the absolute path of a directory of images to be detected. "
    )
    parser.add_argument(
        "--input_annotations",
        help="the absolute path of the annotation file of images. "
    )

    parser.add_argument(
        "--output",
        help="a file path to save output predictions",
        default="./output/output"
    )

    parser.add_argument(
        "--output_format",
        help="format to save the combined detection.",
        default="microsoft tsv"
    )

    parser.add_argument(
        "--nocaps_det_dir",
        help="absolute path of nocaps detector image-feature-extractors",
        default="../image-feature-extractors"
    )

    parser.add_argument(
        "--UniDet_dir",
        help="path of UniDet dir",
        default="../UniDet"
    )

    parser.add_argument(
        "--UniDet_model",
        help="specify which UniDet model you want to use",
        default="Unified_learned_OCI_R50_8x"
    )

    parser.add_argument(
        "--device",
        help="a number specify which GPU you want to use for docker",
        default="0"
    )

    parser.add_argument(
        "--steps",
        help="0: all 3 steps, 1: nocaps detector step only, 2: UniDet step only, 3: combination step only",
        default="0"
    )

    return parser


def process_Nocaps_detector(path_input_imgs, path_input_annotations, path_nocaps_det, device):
    os.system("nvidia-docker run -it --gpus 'device=%s' \
    --name oi_container \
    -v %s/scripts:/workspace/scripts \
    -v %s:/datasets/img \
    -v %s:/datasets/annotations.json \
    oi_image \
    python3 scripts/extract_boxes_oi.py \
    --graph models/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018/frozen_inference_graph.pb \
    --images /datasets/img \
    --annotations /datasets/annotations.json \
    --output /outputs/nocaps_detections.json" % (device, path_nocaps_det, path_input_imgs, path_input_annotations))
    os.system("docker container cp oi_container:/outputs/nocaps_detections.json ./data")
    print("Removing container...")
    os.system("docker rm oi_container")


def process_UniDet_detector(path_input_imgs, UniDet_model, path_UniDet):
    pwd = os.getcwd()
    os.chdir(path_UniDet)
    os.system("python projects/UniDet/demo/demo.py \
    --config-file projects/UniDet/configs/%s.yaml \
    --input %s \
    --output_json %s/data/UniDet_predictions.json \
    --opts MODEL.WEIGHTS \
    models/%s.pth" % (UniDet_model, path_input_imgs, pwd, UniDet_model))
    os.chdir(pwd)


def combine_results(format, output_file_path):
    path_nocaps_det = "./data/nocaps_detections.json"
    path_UniDet_det = "./data/UniDet_predictions.json"
    path_sublist = "./data/sub_list.json"
    if format == "microsoft tsv":
        output_result_path = output_file_path + '.tsv'
        no_output_path = output_file_path + '_noouput.txt'
        # read data
        with open(path_sublist, 'r') as f:
            sub_list = json.load(f)
        with open(path_nocaps_det, 'r') as f:
            nocaps_det_dict = json.load(f)
        with open(path_UniDet_det, 'r') as f:
            detection_dict = json.load(f)
            predictions = detection_dict['predictions']
        # result dictionary
        result_dict = {}
        nooutput_dict = {}
        # process nocaps det data
        categories_list = nocaps_det_dict['categories']
        images_list = nocaps_det_dict['images']
        annotations_list = nocaps_det_dict['annotations']
        category_dict = {}
        image_dict = {}
        imagehw_dict = {}
        for ele in categories_list:
            category_dict.update({ele['id']:ele['name']})
        for ele in images_list:
            image_dict.update({ele['id']:ele['file_name'].split('.')[0]})
            imagehw_dict.update({ele['id']:{'height': ele['height'], 'width': ele['width']}})
            result_dict.update({ele['file_name'].split('.')[0]:[]})
            nooutput_dict.update({ele['file_name'].split('.')[0]:1})
        for ele in annotations_list:
            name = category_dict[ele['category_id']].capitalize()
            if name == 'Horn':
                name = 'French horn'
            elif name == 'Sunflower':
                name = 'Common sunflower'
            elif name == 'Lifejacket':
                name = 'Personal flotation device'
            elif name == 'Dairy':
                name = 'Dairy Product'
            elif name == 'Asparagus':
                name = 'Garden Asparagus'
            if sub_list.get(name) != None:
            # nocaps detector
                image_id = image_dict[ele['image_id']]
                width = imagehw_dict[ele['image_id']]['width']
                height = imagehw_dict[ele['image_id']]['height']
                # label_name = str2mid_dict[name]
                score = ele['score']
                xmin = ele['bbox'][0]
                xmax = ele['bbox'][2]
                ymin = ele['bbox'][1]
                ymax = ele['bbox'][3]
                item = {
                    'class':name,
                    'conf':score,
                    'rect':[xmin, xmax, ymin, ymax]
                }
                result_dict[image_id].append(item)
                if nooutput_dict.get(image_id) != None:
                    nooutput_dict.pop(image_id)
        # process UniDet data
        nocaps_det_nooutput = {}
        for ele in predictions:
            image_id = ele['image_id']
            if len(result_dict[image_id]) == 0:
                nocaps_det_nooutput.update({image_id:1})
            width = ele['width']
            height = ele['height']
            score_list = ele['scores']
            bbox_list = ele['bboxes']
            label_list = ele['labels']
            for i in range(len(bbox_list)):
                name = label_list[i]
                score = score_list[i]
                if name == 'Horn':
                    name = 'French horn'
                elif name == 'Sunflower':
                    name = 'Common sunflower'
                elif name == 'Lifejacket':
                    name = 'Personal flotation device'
                elif name == 'Dairy':
                    name = 'Dairy Product'
                elif name == 'Asparagus':
                    name = 'Garden Asparagus'
                elif name == 'Mouse1':
                    name = 'Mouse'
                elif name == 'Computer Mouse':
                    name = 'Computer mouse'
                elif name == 'Mouse2':
                    name = 'Mouse'
                elif name == 'Bench1':
                    name = 'Bench'
                elif name == 'Bench2':
                    name = 'Bench'
                elif name == 'Toiletries':
                    name = 'Cosmetics'
                elif name == 'Wild bird':
                    name = 'Magpie'
                elif name == 'Frisbee':
                    name = 'Flying disc'
                elif name == 'Hamimelon':
                    name = 'Cantaloupe'
                elif name == 'Remote':
                    name = 'Remote control'
                if sub_list.get(name) == None or nocaps_det_nooutput.get(image_id) != None:
                    # UniDet
                    if len(name) == 0:
                        continue
                    # if str2mid_dict.get(name) == None:
                    #     continue
                    # label_name = str2mid_dict[name]
                    xmin = bbox_list[i][0]
                    xmax = bbox_list[i][2]
                    ymin = bbox_list[i][1]
                    ymax = bbox_list[i][3]
                    item = {
                        'class':name,
                        'conf':score,
                        'rect':[xmin, xmax, ymin, ymax]
                    }
                    result_dict[image_id].append(item)
                    if nooutput_dict.get(image_id) != None:
                        nooutput_dict.pop(image_id)
        
        Unidet_nooutput = {}
        for ele in annotations_list:
            name = category_dict[ele['category_id']].capitalize()
            if name == 'Horn':
                name = 'French horn'
            elif name == 'Sunflower':
                name = 'Common sunflower'
            elif name == 'Lifejacket':
                name = 'Personal flotation device'
            elif name == 'Dairy':
                name = 'Dairy Product'
            elif name == 'Asparagus':
                name = 'Garden Asparagus'
            image_id = image_dict[ele['image_id']]
            if len(result_dict[image_id]) == 0:
                Unidet_nooutput.update({image_id:1})
            if Unidet_nooutput.get(image_id) != None:
                # nocaps detector
                width = imagehw_dict[ele['image_id']]['width']
                height = imagehw_dict[ele['image_id']]['height']
                # label_name = str2mid_dict[name]
                score = ele['score']
                xmin = ele['bbox'][0]
                xmax = ele['bbox'][2]
                ymin = ele['bbox'][1]
                ymax = ele['bbox'][3]
                item = {
                    'class':name,
                    'conf':score,
                    'rect':[xmin, xmax, ymin, ymax]
                }
                result_dict[image_id].append(item)
                if nooutput_dict.get(image_id) != None:
                    nooutput_dict.pop(image_id)

        # output to tsv file
        with open(output_result_path, 'w') as csvf:
            tsv_writer = csv.writer(csvf, delimiter='\t')
            for k,v in result_dict.items():
                tsv_writer.writerow([k, v])
        with open(no_output_path, 'w') as f:
            for k,v in nooutput_dict.items():
                f.write(str(k)+'\n')
    else:
        print("Format not supported")
        exit(-1)


if __name__ == "__main__":
    args = get_parser().parse_args()
    input_imgs = os.path.abspath(args.input_imgs)
    input_annotations = os.path.abspath(args.input_annotations)
    output_file_path = args.output
    output_format = args.output_format
    path_nocaps_det = os.path.abspath(args.nocaps_det_dir)
    path_UniDet = args.UniDet_dir
    UniDet_model = args.UniDet_model
    device = args.device
    steps = args.steps
    print("Arguments: " + str(args))
    if steps == "0" or steps == "1":
        print("Processing Images Using Nocaps Detector...")
        process_Nocaps_detector(input_imgs, input_annotations, path_nocaps_det, device)
    if steps == "0" or steps == "2":
        print("Processing Images Using UniDet Detector...")
        process_UniDet_detector(input_imgs, UniDet_model, path_UniDet)
    if steps == "0" or steps == "3":
        print("Combining Results...")
        combine_results(output_format, output_file_path)
    print("Done.")