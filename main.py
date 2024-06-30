import torch
from DanbooruTagger import *
from PIL import Image
from torchvision.transforms import Resize,ToTensor,Pad
from datasketch import WeightedMinHashGenerator,MinHashLSHForest
import glob
import os
from flask import Flask, request, render_template
import base64
import numpy as np 

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)
image_folder = "G:\comfyui\Blender_ComfyUI\ComfyUI\output"

model=torch.load('tagImg3_ls.pt')
model=torch.nn.Sequential(*(list(model.base_model.children())[:-1]))
model.to(device)
transf = ToTensor()
model.eval()
print(model)

def img2vec(model,image_file):
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    maxlen=img.width if img.width > img.height else img.height
    img=Pad([int((maxlen-img.width)/2),int((maxlen-img.height)/2)],fill=(0,0,0),padding_mode='constant')(img)
    inputs=Resize((256,256))(img)
    inputs=transf(inputs).resize_([1,3,inputs.height,inputs.width])
    with torch.no_grad():
        inputs=inputs.to(device)
        outputs = model(inputs)
    torch.cuda.empty_cache()
    return outputs.cpu().numpy().reshape(-1)

forest = MinHashLSHForest(num_perm=200,l=3)
mg = WeightedMinHashGenerator(1280, 200)
# 添加向量及其ID到LSH Forest和字典
image_files = glob.glob(os.path.join(image_folder, "*.png"))
n=0
for vector_id in image_files:
    m = mg.minhash(img2vec(model, vector_id))
    forest.add(vector_id,m)
    n+=1
forest.index()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('t.html')

#该函数获取一张图片，并且返回最接近上传图片的10张图片的文件名
@app.route('/img_query', methods=['POST'])
def img_infer():
    global forest,mg
    image_file = request.files['image']
    img_vec=img2vec(model, image_file)
    target_vector=mg.minhash(img_vec)
    return forest.query(target_vector, 10)

#该函数获取一张图片的文件名，并且返回对应的图片
@app.route('/photo', methods=['POST'])
def img_get():
    name=request.values.get('key')
    with open(name, 'rb') as f:
        res = base64.b64encode(f.read())
    return res

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005)
