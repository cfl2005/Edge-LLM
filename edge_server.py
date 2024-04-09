import os
import platform
import signal
import readline
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim
import lora
from typing import Optional
from torch.nn import CrossEntropyLoss
import datetime
import pandas as pd
import math
import subprocess
import time
import gc
import sys
import numpy as np
import argparse

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class Int8():
    def __int__(self):
        self.scale = None
        pass

    def tensor_int8(self, tensor):
        tensor = tensor.to(torch.float32).to('cpu')
        max_value = torch.max(tensor)
        min_value = torch.min(tensor)
        self.scale = (max_value - min_value) / 255
        tensor_int8 = torch.quantize_per_tensor(tensor, self.scale, 0, dtype=torch.qint8)
        tensor_int8 = np.array(tensor_int8.int_repr())
        tensor_int8 = torch.tensor(tensor_int8).to(torch.int8)
        return tensor_int8

    def list_int8(self, list_of_tensor):
        for i in range(len(list_of_tensor)):
            list_of_tensor[i] = self.tensor_int8(list_of_tensor[i])
        return list_of_tensor

    def model_int8(self, model):
        for param in model.parameters():
            param.requires_grad = False
            param.data = self.tensor_int8(param.data)
        return model


class GLM_Model():
    def __init__(self, model_name_or_path):

        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True).cuda()
        self.model.eval()

    def build_prompt(self, data):
        inputs_list = []
        for item in data:
            item_query = item
            inputs = self.tokenizer.build_prompt(item_query, history=[])
            inputs_list.append(inputs)
        return inputs_list

    def predict(self, data):
        history = None, []
        data = self.build_prompt(data)
        ret = self.model.chat(self.tokenizer, query=data, history=history)
        return ret  # list from glm


class Lora_base(nn.Module):
    def __init__(self):
        super(Lora_base, self).__init__()
        self.layers = nn.ModuleList(
            [lora.layers.Linear(in_features=4096, out_features=4096, r=2, lora_alpha=1) for _ in range(28)])
        # output_layer 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_layer = nn.Linear(4096, 65024, bias=False,
                                      dtype=torch.float32, device=device)

    def forward(self, list_glm):
        out = 0.5 * list_glm[0] + 0.5 * list_glm[1]
        out = self.layers[0](out)
        for layer, data in zip(self.layers[1:], list_glm[2:]):
            out = 0.5 * out + 0.5 * data
            out = layer(out)
        lm_logits = self.output_layer(out)
        torch_gc()
        lm_logits = lm_logits[-1:]
        lm_logits = lm_logits.transpose(0, 1).contiguous()
        return lm_logits


class Lora_Model_inference():
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.model = Lora_base().half().cuda()

    def load(self):
        if self.model_name_or_path != '':
            self.model.load_state_dict(torch.load(self.model_name_or_path))

    def predict(self, glm_list):
        lm_logits_all_token = []
        grouped_glm = [glm_list[i:i + 29] for i in range(0, len(glm_list), 29)]
        with torch.no_grad():
            for group in grouped_glm:
                lm_logits = self.model(group)
                _, lm_logits_token = torch.max(lm_logits[:, :, :64794], dim=2)
                lm_logits_token = lm_logits_token.item()
                lm_logits_all_token.append(lm_logits_token)
        return lm_logits_all_token


class Lora_Model_train():
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.model = Lora_base().half().cuda()

    def load(self):
        if self.model_name_or_path != '':
            self.model.load_state_dict(torch.load(self.model_name_or_path))

    def compute_loss(self, lm_logits, labels: Optional[torch.Tensor] = None):  # compute loss
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            padding_size = shift_labels.size()[0] - shift_logits.size()[1]
            
            if padding_size > 0:
                padding = torch.zeros(shift_logits.size()[0], padding_size, shift_logits.size()[2]).to(
                    shift_logits.device)
                shift_logits = torch.cat((shift_logits, padding), dim=1)
            elif padding_size < 0:
                padding_tensor = torch.zeros([abs(padding_size)]).to(shift_labels.device)
                shift_labels = torch.cat((shift_labels, padding_tensor), dim=0)
                shift_labels = shift_labels.clone().detach().to(torch.int64)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            torch_gc()
            loss = loss.to(torch.float32)
        return loss

    def train(self, glm_list, pre_label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lm_logits_all_tensor = torch.randn([1, 1, 65024]).to(device).to(torch.float16)
        grouped_glm = [glm_list[i:i + 29] for i in range(0, len(glm_list), 29)]
        for group in grouped_glm:
            lm_logits = self.model(group)
            lm_logits_all_tensor = torch.cat((lm_logits_all_tensor, lm_logits), dim=1)
        lm_logits_all_tensor = lm_logits_all_tensor[..., 1:, :].contiguous()
        labels = torch.tensor(pre_label).to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        loss = self.compute_loss(lm_logits_all_tensor, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class Lora_Model_int8(nn.Module):
    def __init__(self):
        super(Lora_Model_int8, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList(
            [lora.layers.Linear(in_features=4096, out_features=4096, r=2, lora_alpha=1) for _ in range(28)])
        self.output_layer = nn.Linear(4096, 65024, bias=False,
                                      dtype=torch.float32, device='cpu')

    def forward(self, list_glm):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        IN8S = Int8()
        out = 0.5 * list_glm[0] + 0.5 * list_glm[1]
        out = IN8S.tensor_int8(out)
        out = self.layers[0](out, IN8S.scale)

        for layer, data in zip(self.layers[1:], list_glm[2:]):
            out = 0.5 * out + 0.5 * data
            out = IN8S.tensor_int8(out)
            out = layer(out, IN8S.scale)
        lm_logits = self.output_layer(out)
        lm_logits = lm_logits[-1:]
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        return lm_logits


class Lora_Model_Int8_inference():
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.model = Lora_Model_int8().cpu()
        self.Int_8 = Int8()

    def load(self):
        self.model = Lora_Model_int8()
        if self.model_name_or_path != '':
            for para in self.model.parameters():
                para.requires_grad = False
                para.data = para.data.to(torch.int8)
            self.model.load_state_dict(torch.load(self.model_name_or_path))
        else:
            self.model = self.Int_8.model_int8(self.model).to('cpu')

    def predict(self, glm_list):
        glm_list = self.Int_8.list_int8(glm_list)
        lm_logits_all_token = []
        grouped_glm = [glm_list[i:i + 29] for i in range(0, len(glm_list), 29)]
        with torch.no_grad():
            for group in grouped_glm:
                lm_logits = self.model(group)
                _, lm_logits_token = torch.max(lm_logits[:, :, :64794], dim=2)
                lm_logits_token = lm_logits_token.item()
                lm_logits_all_token.append(lm_logits_token)
        return lm_logits_all_token


def task_assignment(sentences, pre_labels, glm_path, inference_path, train_path, int_8_path, inference_tag, train_tag,
                    int_8_tag):
    GLM = GLM_Model(glm_path)
    GLM.load()

    if inference_tag != 0:
        Model_inference = Lora_Model_inference(inference_path)
        Model_inference.load()
    if train_tag != 0:
        Model_train = Lora_Model_train(train_path)
        Model_train.load()
    if int_8_tag != 0:
        Model_int8 = Lora_Model_Int8_inference(int_8_path)
        Model_int8.load()
    for sentence, pre_label in zip(sentences, pre_labels):
        with torch.no_grad():
            glm_list = GLM.predict(sentence)
        if isinstance(pre_label, str):
            if train_tag != 0:
                pre_label = GLM.tokenizer.encode(pre_label, add_special_tokens=False)
                Model_train.train(glm_list, pre_label)
            else:
                print("please choose a train model")
        else:
            if int_8_tag * inference_tag == 0:
                if inference_tag != 0 and int_8_tag == 0:
                    token=Model_inference.predict(glm_list)
                elif int_8_tag != 0 and inference_tag == 0:
                    token=Model_int8.predict(glm_list)
                else:
                    print("please choose a inference model")
            else:
                print("can't use two models to inference")


def main():
    sentences = [['hi'], ['hello']]
    pre_labels = [None, None]
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--glm_path', type=str, default="../model/chatglm2-6b", help='The path of ChatGLM')
    parser.add_argument('--inference_path', type=str, default='', help='The path of lora_inference')
    parser.add_argument('--train_path', type=str, default='', help='The path of lora_train')
    parser.add_argument('--int_8_path', type=str, default='', help='The path of lora_inference_int8')
    parser.add_argument('--inference_tag', type=int, default=0,
                        help='Whether to use inference mode 0 means not using it 1 means using it ')
    parser.add_argument('--train_tag', type=int, default=0,
                        help='Whether to use train mode 0 means not using it 1 means using it')
    parser.add_argument('--int_8_tag', type=int, default=0,
                        help='Whether to use inference mode of int8 0 means not using it 1 means using it')
    args = parser.parse_args()
    task_assignment(sentences, pre_labels, args.glm_path, args.inference_path, args.train_path, args.int_8_path,
                    args.inference_tag, args.train_tag, args.int_8_tag)


if __name__ == "__main__":
    main()
, pre_labels, args.glm_path, args.inference_path, args.train_path, args.int_8_path,
                    args.inference_tag, args.train_tag, args.int_8_tag)


if __name__ == "__main__":
    main()
