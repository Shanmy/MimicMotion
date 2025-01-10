from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor

from PIL import Image

def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray, 
        crop_with_offset: bool=False,
        sample_stride: int=1):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # compute linear-rescale params [???] TODO: understand this
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    pix_offset = []
    # pose rescale 
    for i, detected_pose in enumerate(detected_poses):
        if crop_with_offset:
            detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
            offset = detected_pose['bodies']['candidate'].mean(axis=0) - 0.5
            detected_pose['faces'] = detected_pose['faces'] * a + b - offset
            detected_pose['hands'] = detected_pose['hands'] * a + b - offset
            detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] - offset
            pix_offset.append([a[0], a[1], (b - offset)[0], (b - offset)[1]])

        im = draw_pose(detected_pose, height, width)
        # Image.fromarray(im.transpose(1, 2, 0)).save(f"/gscratch/realitylab/shanmy/MimicMotion/assets/draft/{i:03d}.png")
        # Image.fromarray(frames[i]).save(f"/gscratch/realitylab/shanmy/MimicMotion/assets/draft/{i:03d}_frame.png")
        output_pose.append(np.array(im))
    return np.stack(output_pose), np.array(pix_offset)


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)
