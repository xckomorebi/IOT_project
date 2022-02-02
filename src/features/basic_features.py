import os

from argparse import Namespace

from VideoPose3D.inference import infer_video_d2


def extract_feature_from_path(inputs, outputs):
    CFG_FILE = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
    IMAGE_EXT = "mp4"

    if not os.path.exists(outputs):
        os.mkdir(outputs)

        for category in os.listdir(inputs):
            input_category = os.path.join(inputs, category)
            output_category = os.path.join(outputs, category)
            ns = Namespace(im_or_folder=input_category,
                           output_dir=output_category,
                           cfg=CFG_FILE,
                           image_ext=IMAGE_EXT)
            infer_video_d2
