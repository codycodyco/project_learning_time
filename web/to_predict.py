import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from learning.t_model import efficientnetv2_s as create_model

def to_predict():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        img_size = {"s": [300, 384],  # train_size, val_size
                    "m": [384, 480],
                   "l": [384, 480]}
        num_model = "s"

        data_transform = transforms.Compose(
            [transforms.Resize(img_size[num_model][1]),
             transforms.CenterCrop(img_size[num_model][1]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

        # load image
        img_path = r"static/to_pridict/to_pridect.png"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = '../learning/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_model(num_classes=5).to(device)
        # load model weights
        model_weight_path = "../learning/weights/model-29.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
        plt.title(print_res)
        s=""
        for i in range(2):
            s+="class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy())
        return s
print("???")