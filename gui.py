import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image, ImageOps
import os
import torch
import json
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision

# model declaration
# Reference: https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation/blob/master/models/modules/StackedHourGlass.py

class BnReluConv(nn.Module):
		"""docstring for BnReluConv"""
		def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
				super(BnReluConv, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.kernelSize = kernelSize
				self.stride = stride
				self.padding = padding

				self.bn = nn.BatchNorm2d(self.inChannels)
				self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
				self.relu = nn.ReLU()

		def forward(self, x):
				x = self.bn(x)
				x = self.relu(x)
				x = self.conv(x)
				return x


class ConvBlock(nn.Module):
		"""docstring for ConvBlock"""
		def __init__(self, inChannels, outChannels):
				super(ConvBlock, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.outChannelsby2 = outChannels//2

				self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
				self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
				self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

		def forward(self, x):
				x = self.cbr1(x)
				x = self.cbr2(x)
				x = self.cbr3(x)
				return x

class SkipLayer(nn.Module):
		"""docstring for SkipLayer"""
		def __init__(self, inChannels, outChannels):
				super(SkipLayer, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				if (self.inChannels == self.outChannels):
						self.conv = None
				else:
						self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

		def forward(self, x):
				if self.conv is not None:
						x = self.conv(x)
				return x

class Residual(nn.Module):
		"""docstring for Residual"""
		def __init__(self, inChannels, outChannels):
				super(Residual, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.cb = ConvBlock(inChannels, outChannels)
				self.skip = SkipLayer(inChannels, outChannels)

		def forward(self, x):
				out = 0
				out = out + self.cb(x)
				out = out + self.skip(x)
				return out
# Reference: https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation/blob/master/models/StackedHourGlass.py

class myUpsample(nn.Module):
	def __init__(self):
		super(myUpsample, self).__init__()
		pass
	def forward(self, x):
		return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)
        
        
class Hourglass(nn.Module):
    """docstring for Hourglass"""
    def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """

        _skip = []
        for _ in range(self.nModules):
            _skip.append(Residual(self.nChannels, self.nChannels))

        self.skip = nn.Sequential(*_skip)

        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through Residual Module or sequence of Modules
        """

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(Residual(self.nChannels, self.nChannels))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            self.hg = Hourglass(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(Residual(self.nChannels,self.nChannels))

            self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

        """
        Now another Residual Module or sequence of Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(Residual(self.nChannels,self.nChannels))

        self.lowres = nn.Sequential(*_lowres)

        """
        Upsampling Layer (Can we change this??????)
        As per Newell's paper upsamping recommended
        """
        self.up = myUpsample()#nn.Upsample(scale_factor = self.upSampleKernel)


    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions>1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        return out2 + out1


class StackedHourGlass(nn.Module):
	"""docstring for StackedHourGlass"""
	def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
		super(StackedHourGlass, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nJoints = nJoints

		self.start = BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

		self.res1 = Residual(64, 128)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = Residual(128, 128)
		self.res3 = Residual(128, self.nChannels)

		_hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

		for _ in range(self.nStack):
			_hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(Residual(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(BnReluConv(self.nChannels, self.nChannels))
			_chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
			_lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, x):
		x = self.start(x)
		x = self.res1(x)
		x = self.mp(x)
		x = self.res2(x)
		x = self.res3(x)
		out = []

		for i in range(self.nStack):
			x1 = self.hourglass[i](x)
			x1 = self.Residual[i](x1)
			x1 = self.lin1[i](x1)
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			x = x + x1 + self.jointstochan[i](out[i])

		return (out)

class OurStackedHourGlass(nn.Module):
  def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
    super(OurStackedHourGlass, self).__init__()
    self.nChannels = nChannels
    self.nStack = nStack
    self.nModules = nModules
    self.numReductions = numReductions
    self.nJoints = nJoints

    self.first = BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)
    self.second = BnReluConv(64, 64, kernelSize = 3, stride = 1, padding = 1)
    self.third = BnReluConv(64, 128, kernelSize = 1, stride = 1, padding = 0)
    self.mp = nn.MaxPool2d(2, 2)
    self.fourth = BnReluConv(128, 128, kernelSize = 3, stride = 1, padding = 1)
    self.fifth = BnReluConv(128, self.nChannels, kernelSize = 1, stride = 1, padding = 0)

    _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

    for _ in range(self.nStack):
      _hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
      _ResidualModules = []
      for _ in range(self.nModules):
        _ResidualModules.append(Residual(self.nChannels, self.nChannels))
      _ResidualModules = nn.Sequential(*_ResidualModules)
      _Residual.append(_ResidualModules)
      _lin1.append(BnReluConv(self.nChannels, self.nChannels))
      _chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
      _lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
      _jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

    self.hourglass = nn.ModuleList(_hourglass)
    self.Residual = nn.ModuleList(_Residual)
    self.lin1 = nn.ModuleList(_lin1)
    self.chantojoints = nn.ModuleList(_chantojoints)
    self.lin2 = nn.ModuleList(_lin2)
    self.jointstochan = nn.ModuleList(_jointstochan)

  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    x = self.third(x)
    x = self.mp(x)
    x = self.fourth(x)
    x = self.fifth(x)
    out = []

    for i in range(self.nStack):
      x1 = self.hourglass[i](x)
      x1 = self.Residual[i](x1)
      x1 = self.lin1[i](x1)
      out.append(self.chantojoints[i](x1))
      x1 = self.lin2[i](x1)
      x = x + x1 + self.jointstochan[i](out[i])

    return (out)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.stackedHG = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=16)

    def forward(self, x):
        x = self.stackedHG(x)
        return x

# image transforms
class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img

class ToTensor(object):
    def __call__(self, sample):
        image = sample
        image = image.transpose((2, 0, 1))
#         image = image.astype('float32')
        return torch.from_numpy(image)

class SquarePad(object):
  def __init__(self, targetSize):
    self.targetSize = int(targetSize)

  def __call__(self, sample):
    image = sample
    targetSize = self.targetSize

    if max(image.shape) < targetSize:
      image = Image.fromarray(image)
      old_size = image.size

      ratio = float(targetSize)/max(old_size)
      new_size = tuple([int(x*ratio) for x in old_size])

      image = image.resize(new_size, Image.ANTIALIAS)
      image = np.asarray(image)

    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    c, h, w = image.shape

    hp, vp = 0, 0
    if w < targetSize:
      hp = int((targetSize - w) / 2)
    if h < targetSize:
      vp = int((targetSize - h) / 2)
    padding = (hp, hp, vp, vp)
  
    image = F.pad(image, padding, 'constant', 0).numpy()
    image = image.transpose((1, 2, 0))
    return image

# helper function
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=15, marker='.', c='r')
    plt.plot(landmarks[:6, 0], landmarks[:6, 1])
    plt.plot(landmarks[6:10, 0], landmarks[6:10, 1])
    plt.plot(landmarks[10:, 0], landmarks[10:, 1])
    plt.pause(0.001)

# load the trained model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Network().to(device)
model = model.double()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
                 
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Human pose estimation')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):

    plt.figure()
    toImg = transforms.ToPILImage()
    index = 10

    image = io.imread(file_path)
    trf = transforms.Compose([SquarePad(256), Resize(256), ToTensor()])
    image = trf(image)
    Y = model(image.unsqueeze(0).to(device))

    for i in range(len(Y)):
        #inx = np.zeros((16, 2))
        a_list = []
        for a in range(len(Y[i])):
            b_list = []
            for b in range(len(Y[i][a])):
                max = Y[i][a][b].argmax().item()
        
                x = (max // 64) * 4
                y = (max % 64) * 4
        
                b_list.append((x,y))
            a_list.append(b_list)
        max_idx = torch.from_numpy(np.asarray(a_list)).to(device)
        break

    show_landmarks(toImg(image.type(torch.FloatTensor)), (max_idx.squeeze().to("cpu").numpy()))
    plt.show()

    label.configure(foreground='#011638', text="Estimated") 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Estimate",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Human Pose Estimator",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
