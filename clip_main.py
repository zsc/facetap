import glob
import numpy as np

# import cv2
import torch
import onnxruntime as ort
import torchvision.transforms as T

# import clip

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

"""
Baseed on background info here:
https://github.com/openai/CLIP/issues/122
"""

print(ort.get_device())
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

"""
def crop_center(img):
    h, w = img.shape[:2]
    s = min(h, w) // 2
    return img[h // 2 - s: h//2 + s, w // 2 - s : w//2 + s]
"""


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# m, pre = clip.load("ViT-B/32")
# m, pre = clip.load("RN50")
pre = T.Compose(
    [
        T.Resize(256, interpolation=BICUBIC),
        T.CenterCrop(224),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)
# labels = ["wash hands with fingers visible", "unoccluded plastic plate"]
labels = [" " + x for x in ["wash hands", "water sink", "cross finger", "...,.,,."]]
ort_sess = ort.InferenceSession("clip_resnet.onnx")
for fn in sorted(
    glob.glob("/data/facetap/test_data/*.JPG")
    + glob.glob("/data/facetap/test_data/*/*.JPG")
):
    print(fn)
    # dummy_texts = clip.tokenize(labels)
    dummy_texts = torch.tensor(np.load("/home/dev/text.npy"))
    scale = 224
    dummy_image = Image.open(fn)
    # dummy_image = cv2.resize(crop_center(cv2.imread(fn)), (scale, scale))[:,:,::-1]
    # dummy_tensor = pre(torch.tensor(dummy_image.astype('float32').transpose(2, 0, 1).reshape(1, 3, scale, scale)))
    dummy_tensor = pre(dummy_image).unsqueeze(0)  # .to(device)
    # dummy_tensor = pre(dummy_image)
    # print(m.forward(dummy_tensor, dummy_texts)) # Original CLIP result (1)
    # cv2.imwrite("t.jpg", dummy_image)
    # from IPython import embed; embed()
    result = ort_sess.run(
        ["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"],
        {"IMAGE": dummy_tensor.numpy(), "TEXT": dummy_texts.numpy()},
    )
    # from IPython import embed; embed()
    print(result[0][0])
    print(labels[result[0][0].argmax()])  # verify that result is comparable to (1)
