from __future__ import print_function, division
from torchvision import transforms
import argparse
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import onnxruntime
import cv2
import time
import numpy as np 
import torch
import torchvision.transforms as T

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


def GetFileFromThisRootDir(dir,ext = None):
    allfiles = []
    needExtFilter = (ext != None)
    for root,dirs,files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]  
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

def Get_img_paths(path):
    if os.path.splitext(path)[-1] == '.txt':
        with open(path, 'r') as f:
            img_paths = f.read().splitlines()
    else:
        img_paths = GetFileFromThisRootDir(path)
    out = [imgpath.split(' ')[0] for imgpath in img_paths]
    out.sort()
    return out

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, txt_path, input_size):
        self.img_paths = Get_img_paths(txt_path)
        self.input_size = input_size
        self.input_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = Image.open(img_path)
        if img.size[0] != self.input_size:
            img = img.resize((self.input_size,self.input_size), Image.ANTIALIAS)
        img = self.input_transform(img)
        return img, img_path


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        # input_feed = {}
        # input_feed['input_images'] = np.expand_dims(image_numpy[0],0)
        # input_feed['ref_images'] = np.expand_dims(image_numpy[1],0)
        a = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return a


if __name__ == '__main__':

    '''
    超分辨率利用onnx进行推断
    运行环境base
    '''
    parser = argparse.ArgumentParser(description='Train Downscaling Models')
    parser.add_argument('--test_txt', default='./test_img/glint', type=str) 
    parser.add_argument('--batch_size', default=1, type=int) 
    parser.add_argument('--onnx_dir', default='./wyt_SR.onnx', type=str) 
    parser.add_argument('--input_size', default=256, type=int)  #结果保存的文件夹名称
    parser.add_argument('--save_dir', default='./test_img/result_onnx', type=str)  #结果保存的文件夹名称
    opt = parser.parse_args()


    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    #datsset
    test_datasets = customData(opt.test_txt, opt.input_size)
    #dataloader
    test_dataloders = torch.utils.data.DataLoader(test_datasets,
                                            batch_size=opt.batch_size,
                                            num_workers=4,
                                            shuffle=False) 
    #读取onnx模型
    onnx_model = ONNXModel(opt.onnx_dir)

    times = []
    output = []
    i = 1
    for data in tqdm(test_dataloders):
        inputs, path = data
        name = os.path.basename(path[0])
        inputs = inputs.permute(0,2,3,1)
        inputs = inputs.cpu().numpy()
        canon_im = onnx_model.forward(inputs)

        canon_im = canon_im[0][0]
        canon_im = (canon_im + 1.0)/2.0
        canon_im = np.clip(canon_im, 0, 1.0)
        # canon_im = canon_im.transpose(1,2,0)
        canon_im = Image.fromarray(np.uint8(canon_im * 255))


        save_path = os.path.join(opt.save_dir, str(i) + '_'+name)
        canon_im.save(save_path, quality=95)
        i = i + 1
print('finished')


