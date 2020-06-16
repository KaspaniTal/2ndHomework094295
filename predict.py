import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import f1_score


# Usefull functions
def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


class costum_images_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, sorted(os.listdir(self.root_dir))[idx])
        image = io.imread(img_name)
        label = img_name[-5]  # get the label of a given image
        sample = (image, int(label))

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample[0], sample[1]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return (img, label)


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        image = image.transpose((2, 0, 1)).astype(float)
        return (torch.from_numpy(image), torch.tensor(label))


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(8 * 8 * 512, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)  
        out = self.dropout(out)
        out = self.fc(out) 

        return self.softmax(out)


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()
test_dataset = costum_images_dataset(root_dir= args.input_folder, transform=transforms.Compose([Rescale((64,64)),ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle=False, num_workers=2)

# Load model
trained_net = to_gpu(CNN_2())
trained_net.load_state_dict(torch.load('cnn_best_v2.pkl', map_location=lambda storage, loc: storage))

# Predict images
trained_net.eval()
correct = 0
total = 0
y_pred = []
for images, labels in test_loader:
    images = to_gpu(images)
    images = images.float()
    outputs = trained_net(images)
    _, predicted = torch.max(outputs.data, 1)
    listy_pred = predicted.tolist()
    y_pred += listy_pred
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

files = sorted(os.listdir(args.input_folder))
prediction_df = pd.DataFrame(zip(files, y_pred), columns=['id', 'label'])
prediction_df.to_csv("prediction.csv", index=False, header=False)

y_true = prediction_df['id'].apply(lambda x: int(x[7:8])).values
f1 = f1_score(y_true, y_pred, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.
print("F1 Score is: {:.2f}".format(f1))


