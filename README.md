Code Reproduction for LLM:
Content includes:
- Environment dependencies
- Data Processing
- Run the code
   1. LLM Training
   2. LLM Inference
   3. Use two lora models
   4. The batch inference of ChatGLM

Requirements:
```
pip3 install -r requirements.txt
```

Data Processing:
Our system's query (corresponding to "sentences" in edge_server.py) and label (corresponding to "pre_labels" in edge_server.py) should have the following format:
```
sentences = [['hello'], ['hi']]
pre_labels = ['yes', None]
```
When you choose to use the training mode, pre_labels cannot be None.

Run the code:
1. Before calling our code, you need to download the model parameter file for Chatglm2-6b locally:
   1. `git clone https://huggingface.co/THUDM/chatglm2-6b`
   2. Manually download from Tsinghua Cloud: [https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/](https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/)
2. Replace the `modeling_chatglm.py` in the downloaded model parameter file with `model/modeling_chatglm.py`.
3. You can call our code using the following steps:
   ```
   1. LLM Training:
      python edge_server.py --glm_path 'your path of ChatGLM2-6b' --train_path 'your path of Lora_train' --train_tag 1
   2. LLM Inference (using fp16 as an example):
      python edge_server.py --glm_path 'your path of ChatGLM2-6b' --inference_path 'your path of Lora_inference' --inference_tag 1
   3. We support loading one Lora for inference and one Lora for training simultaneously (using fp16 as an example):
      python edge_server.py --glm_path 'your path of ChatGLM2-6b' --train_path 'your path of Lora_train' --train_tag 1 --inference_path 'your path of Lora_inference' --inference_tag 1
   4. The batch inference of ChatGLM (using batchsize is 2 as an example):
      python edge_server.py --glm_path 'your path of ChatGLM2-6b' --bs 2
      Note:
         you sentences should have the following format:
            sentences = [['hi','nice'], ['hello','good']]
   ```

Please note that you need to replace "your path of ChatGLM2-6b", "your path of Lora_train", and "your path of Lora_inference" with the actual paths on your system. In this case, you can set "your path of Lora_train" and "your path of Lora_inference" to empty strings, which will initialize a new Lora model.
