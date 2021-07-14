import os
import cv2
import torch
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
parser = ArgumentParser(description='generate depth map with MiDaS')
parser.add_argument('--model_type',default='l',type=str,help='one of {l,m,s}')
parser.add_argument('--input_dir',default='data/gt',type=str,help='path of input images')
parser.add_argument('--output_dir',default='data/depth',type=str,help='path of output depth maps')
parser.add_argument('--sort_by_num',default=True,type=bool,help='whether sort the image names by their numbers or not')
args = parser.parse_args()
# MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
types = {'l':"DPT_Large",'m':"DPT_Hybrid",'s':"MiDaS_small"}
model_type = types[args.model_type]
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def depthPrediction(imgName):
    img = cv2.imread(imgName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    return output
imgDir = args.input_dir
resDir = args.output_dir
if resDir=="data/depth":
    resDir = imgDir.replace(imgDir.split('/')[-1],'depth')
os.makedirs(resDir,exist_ok=True)
names = os.listdir(imgDir)
if args.sort_by_num:
    names.sort(key=lambda x:int(x.split('.')[0]))
for name in names:
    imgName = os.path.join(imgDir,name)
    res = depthPrediction(imgName)
    plt.imsave(os.path.join(resDir,name),res)
    print(name)
