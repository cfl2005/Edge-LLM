Code Reproduction for LLM:
## Content includes:
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

## Data Processing:
Our system's query (corresponding to "sentences" in edge_server.py) and label (corresponding to "pre_labels" in edge_server.py) should have the following format:
```
sentences = [['hello'], ['hi']]
pre_labels = ['yes', None]
```
When you choose to use the training mode, pre_labels cannot be None.

## Run the code:
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


## Scheduler Data Input
The data input is a list containing multiple dictionaries, with each dictionary representing a task.

For example, each dictionary in the `datas` list should have the following structure:

```python
{
    'sentence': 'task1',    # Task description
    'data_length': 1,       # Data length
    'start_time': 0,        # Task start time (must be 0)
    'true_time': 3          # Expected end time of the task
}
```

## Functionality

### `task_process(datas, interval_time, batch_size)`

Preprocesses the input dataset.

- `datas`: Dataset
- `interval_time`: Interval time
- `batch_size`: Number of tasks to be sent within a time interval

### `task_do(tasks_processed, current_time)`

Prepares the tasks that should be predicted at the current time.

- `tasks_processed`: List of tasks after initial processing
- `current_time`: Current time

### `task_sjf(tasks_processed, current_time, batch_size)`

Sorts the tasks using the Shortest Job First (SJF) scheduling algorithm.

- `tasks_processed`: List of tasks after initial processing
- `current_time`: Current time
- `batch_size`: Number of tasks to be predicted at once

### `task_edf(tasks_processed, current_time, batch_size)`

Sorts the tasks using the Earliest Deadline First (EDF) scheduling algorithm.

- `tasks_processed`: List of tasks after initial processing
- `current_time`: Current time
- `batch_size`: Number of tasks to be predicted at once

### `glm_predict(model, tasks_processed, schedule, batch_size)`

Uses the model to predict the tasks.

- `model`: Model

if you want run this program,for exmple:
- python schedule.py --task_batch 2 --batch_size_predict 16 --schedule_mode sjf --interval_time 0.1
