import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from ai.model.mobilenetv3 import MobileNetV3
from ai.model.pix2pix import Generator

class Inference:
    def __init__(self, c_weight=None, g_weight=None, num_classes=28):
        
        self.classification_model = MobileNetV3(num_classes=num_classes).cpu()
        self.classification_model.load_state_dict(torch.load(c_weight, map_location=torch.device('cpu')))
        self.classification_model.eval()
        
        self.generation_model = Generator().cpu()
        self.generation_model.load_state_dict(torch.load(g_weight, map_location=torch.device('cpu')))
        self.generation_model.eval()
        
        self.classes = {
            0: '얼레지',
            1: '노루귀',
            2: '애기똥풀',
            3: '제비꽃',
            4: '민들레',
            5: '할미꽃',
            6: '은방울꽃',
            7: '비비추',
            8: '패랭이꽃',
            9: '수련',
            10: '맥문동',
            11: '엉겅퀴',
            12: '참나리',
            13: '초롱꽃',
            14: '상사화',
            15: '동백',
            16: '개망초',
            17: '장미',
            18: '해바라기',
            19: '무궁화',
            20: '진달래',
            21: '개나리',
            22: '수국',
            23: '연꽃',
            24: '나팔꽃',
            25: '목련',
            26: '벚꽃',
            27: '튤립',
        }
        
    @torch.no_grad()
    def classification(self, image):
        inputs = self.load_image(image)
        output = self.classification_model(inputs)
        prob_with_idx = torch.sort(F.softmax(output))
        result = []
        total = prob_with_idx[0][0][-3:].sum().item()
        for i in range(1, 4):
            prob = prob_with_idx[0][0][-3:][-i].item()
            idx = prob_with_idx[1][0][-3:][-i].item()
            prob = f"{int((prob / total) * 100)}%"
            output = {
                'probability': prob,
                'type': self.classes[idx]
            }
            result.append(output)
        return result
    
    @torch.no_grad()
    def generation(self, image):
        inputs = self.load_image(image)
        output = self.generation_model(inputs)
        return output
        
    def load_image(self, path):
        img = Image.open(path)
        img = img.resize((256, 256))
        img = np.array(img) / 255.
        img = torch.Tensor(img).permute(2,0,1)
        return img.unsqueeze(dim=0)

c_weight_path = './ai/weight/mobilenetv3_weight.pt'
c_inference = Inference(c_weight=c_weight_path)

def classify(image_src):
    return c_inference.classification(image_src)

g_weight_path = './ai/weight/generation_model.pt'
g_inference = Inference(g_weight=g_weight_path)

def generate(image_src):
    return g_inference.generation(image_src)