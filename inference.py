import cv2

import torch
import torch.nn.functional as F

from model.classification_model import VGG19
from model.generation_model import Generator

class Inference:    
    def __init__(self, c_weight=None, g_weight=None, num_classes=37):
        self.classification_model = VGG19(num_classes=num_classes).cpu()
        self.classification_model.load_state_dict(torch.load(c_weight, map_location=torch.device('cpu')))
        self.generation_model = Generator().cpu()
        self.generation_model.load_state_dict(torch.load(g_weight, map_location=torch.device('cpu')))
        self.classes = {
            0: '얼레지',
            1: '노루귀',
            2: '애기똥풀',
            3: '제비꽃',
            4: '민들레',
            5: '붓꽃',
            6: '할미꽃',
            7: '깽깽이풀',
            8 : '삼지구엽초',
            9 : '현호색',
            10: '은방울꽃',
            11: '복수초',
    
            12: '비비추',
            13: '동자꽃',
            14: '곰취',
            15: '패랭이꽃',
            16: '약모밀',
            17: '닭의장풀',
            18: '수련',
            19: '맥문동',
            20: '물봉선',
            21: '엉겅퀴',
            22: '참나리',
            23: '노루오줌',
            
            24: '구절초',
            25: '꿩의비름',
            26: '투구꽃',
            27: '참취',
            28: '용담',
            29: '마타리',
            30: '국화',
            31: '쑥부쟁이',
            32: '초롱꽃',
            33: '과꽃',
            34: '상사화',
            
            35: '동백',
            36: '솜다리',
        }
        
    @torch.no_grad()
    def classify(self, image):
        inputs = self.load_image(image)
        output = self.classification_model(inputs)
        prob_with_idx = torch.sort(F.softmax(output, dim=1))
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
        img = cv2.imread(path)
        img = cv2.cvtColor(img)
        img = cv2.resize(img, (256, 256))
        img = torch.Tensor(img).permute(2,0,1)
        return img.unsqueeze(dim=0)

c_weight_path = './ai/weight/classification_model.pt'
c_inference = Inference(c_weight=c_weight_path)

def classify(image_src):
    return c_inference.classification(image_src)

g_weight_path = './ai/weight/generation_model.pt'
g_inference = Inference(g_weight=g_weight_path)

def generate(image_src):
    return g_inference.generation(image_src)