import argparse
import os
from tqdm import tqdm
from PIL import Image
import onnxruntime
import time
import numpy as np 
import torch
import torchvision.transforms as T
from facenet_pytorch.models.mtcnn import MTCNN
from align_faces import align_face

detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device='cpu')

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
        #
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        # input_feed = {}
        # input_feed['input_images'] = np.expand_dims(image_numpy[0],0)
        # input_feed['ref_images'] = np.expand_dims(image_numpy[1],0)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output


if __name__ == '__main__':

    '''
    超分辨率利用onnx进行推断(envir:base)
    '''
    parser = argparse.ArgumentParser(description='face super resolution')
    parser.add_argument('--test_txt', default='./img', type=str) 
    parser.add_argument('--batch_size', default=1, type=int) 
    parser.add_argument('--onnx_dir', default='./wyt_SR.onnx', type=str) 
    parser.add_argument('--save_dir', default='./result', type=str)  
    opt = parser.parse_args()

    #read onnx
    onnx_model = ONNXModel(opt.onnx_dir)
    input_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    img_paths = Get_img_paths(opt.test_txt)

    for img_path in tqdm(img_paths):
        name = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path)

        #detect face and landmark
        _, _, landmarks = detector.detect(img, landmarks=True) 

        if landmarks is None: continue
        num_faces = landmarks.shape[0]
        for j in range(num_faces):
            lr = align_face(img, landmarks[j])

            #data preprocess
            inputs = input_transform(lr)
            inputs = inputs.unsqueeze(0).permute(0,2,3,1)
            inputs = inputs.cpu().numpy()

            #inference
            sr = onnx_model.forward(inputs)
            sr = sr[0][0]
            sr = (sr + 1.0)/2.0
            sr = np.clip(sr, 0, 1.0)
            sr = Image.fromarray(np.uint8(sr * 255))

            #save
            save_path = os.path.join(opt.save_dir, name + '_face_' + str(j).zfill(3) + '.png' )
            sr.save(save_path, quality=95)

    print('finish inference!')
