import glob
import cv2
import torch
import onnxruntime as ort
import clip

'''
Baseed on background info here:
https://github.com/openai/CLIP/issues/122
'''

def crop_center(img):
    h, w = img.shape[:2]
    s = min(h, w) // 2
    return img[h // 2 - s: h//2 + s, w // 2 - s : w//2 + s]

#m, pre = clip.load("ViT-B/32")
m, pre = clip.load("RN50")
#labels = ["wash hands with fingers visible", "unoccluded plastic plate"]
labels = [" " + x for x in ["wash hands", "water sink", "cross finger", "...,.,,."]]
for fn in sorted(glob.glob("/data/facetap/test_data/*.JPG") + glob.glob("/data/facetap/test_data/*/*.JPG")):
    print(fn)
    dummy_texts = clip.tokenize(labels)
    scale = 224
    dummy_image = cv2.resize(crop_center(cv2.imread(fn)), (scale, scale))[:,:,::-1]
    dummy_tensor = torch.tensor(dummy_image.astype('float32').transpose(2, 0, 1).reshape(1, 3, scale, scale))
    #print(m.forward(dummy_tensor, dummy_texts)) # Original CLIP result (1)
    cv2.imwrite("t.jpg", dummy_image)
    ort_sess = ort.InferenceSession("clip_resnet.onnx")
    result=ort_sess.run(["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"], {"IMAGE": dummy_tensor.numpy(), "TEXT": dummy_texts.numpy()})
    #from IPython import embed; embed()
    print(result[0][0])
    print(labels[result[0][0].argmax()]) # verify that result is comparable to (1)
