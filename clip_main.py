import glob
import numpy as np

# import cv2
import torch
import onnxruntime as ort
import torchvision.transforms as T

# import clip

from PIL import Image


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

# m, pre = clip.load("ViT-B/32")
# m, pre = clip.load("RN50")
pre = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
