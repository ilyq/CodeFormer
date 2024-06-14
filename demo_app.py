import os
import cv2

import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url

from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
}


def get_optimal_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device(gpu_id=None):
    if gpu_id is None:
        gpu_str = ""
    elif isinstance(gpu_id, int):
        gpu_str = f":{gpu_id}"
    else:
        raise TypeError("Input should be int value.")

    return torch.device(
        "cuda" + gpu_str
        if torch.cuda.is_available() and torch.backends.cudnn.is_available()
        else "cpu"
    )


def set_realesrgan(*, bg_tile: int, model_name: str = "RealESRGAN_x4plus"):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer

    if model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    elif model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    elif model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=file_url,
        model=model,
        tile=bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=True,
    )

    return upsampler


def image_enhance(
    *,
    input_path: str,
    output_path: str,
    fidelity_weight: float,
    upscale: int,
    detection_model: str = "retinaface_resnet50",
    bg_upsampler: str = "realesrgan",
    face_upsample: bool = True,
    bg_tile: int = 400,
):
    """
    图片增强
    :param input_path:
    :param output_path:
    :param fidelity_weight: Balance the quality and fidelity. Default: 0.5
    :param upscale: The final upsampling scale of the image. Default: 2
    :param detection_model: Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. Default: retinaface_resnet50
    :param bg_upsampler: Background upsampler. Optional: realesrgan
    :param face_upsample: Face upsampler after enhancement. Default: False
    :param bg_tile: Tile size for background sampler. Default: 400
    :return:
    """
    device = get_device()

    w = fidelity_weight

    # ------------------ set up background upsampler ------------------
    if bg_upsampler == "realesrgan":
        bg_upsampler = set_realesrgan(bg_tile=bg_tile)
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(bg_tile=bg_tile)
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(
        url=pretrain_model_url["restoration"],
        model_dir="weights/CodeFormer",
        progress=True,
        file_name=None,
    )
    checkpoint = torch.load(ckpt_path)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()

    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )

    # -------------------- start to processing ---------------------
    # for i, img_path in enumerate(input_img_list):
    # clean all the intermediate results to process the next image
    face_helper.clean_all()

    if isinstance(input_path, str):
        basename, _ = os.path.splitext(input_path)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    face_helper.read_image(img)
    # get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    print(f"\tdetect {num_det_faces} faces")
    # align and warp each face
    face_helper.align_warp_face()

    # face restoration for each cropped face
    for _, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"\tFailed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)

    # upsample the background
    if bg_upsampler is not None:
        # Now only support RealESRGAN for upsampling background
        bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
    else:
        bg_img = None

    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    if face_upsample and face_upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=False,
            face_upsampler=face_upsampler,
        )
    else:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=False
        )

    # save restored img
    if restored_img is not None:
        save_restore_path = os.path.join(output_path, f"{basename}.png")
        imwrite(restored_img, save_restore_path)

    print(f"\nAll results are saved in {output_path}")


if __name__ == "__main__":
    image_enhance(
        # input_path="01.png",
        # input_path="02.jpg",
        input_path="05_1.jpg",
        output_path="results",
        fidelity_weight=0.5,
        upscale=2,
    )
