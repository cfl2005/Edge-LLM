import random
import time
import argparse
from edge_server import *


def task_process(tasks, interval_time, batch_size):
    for i, task in enumerate(tasks):
        group_number = i // batch_size
        task['start_time'] = group_number * interval_time
        task['true_time'] += group_number * interval_time
    return tasks


def generate_random_tasks(num_tasks):
    tasks = []

    for i in range(0, num_tasks):
        data_length = random.randint(1, 10)
        true_time = random.randint(data_length, 20)
        task = {
            'sentence': f'task{i}',
            'data_length': data_length,
            'start_time': 0,
            'true_time': true_time
        }
        tasks.append(task)

    return tasks


def task_do(tasks_processed, current_time):
    tasks_to_do = [task for task in tasks_processed if task['start_time'] <= current_time]
    return tasks_to_do


def task_sjf(tasks_processed, current_time, batch_size):
    tasks_to_do = task_do(tasks_processed, current_time)
    sorted_tasks = sorted(tasks_to_do, key=lambda x: x['data_length'])
    current_over_tasks = sorted_tasks[:batch_size]
    for task in current_over_tasks:
        tasks_processed.remove(task)
    return current_over_tasks, tasks_processed


def task_edf(tasks_processed, current_time, batch_size):
    tasks_to_do = task_do(tasks_processed, current_time)
    sorted_tasks = sorted(tasks_to_do, key=lambda x: x['true_time'])
    current_over_tasks = sorted_tasks[:batch_size]
    for task in current_over_tasks:
        tasks_processed.remove(task)
    return current_over_tasks, tasks_processed


def glm_predict(model, tasks_processed, schedule, batch_size):
    current_time = 0
    processed_tasks = []
    model.load()
    while tasks_processed:
        if schedule == 'sjf':
            current_over_tasks, tasks_processed = task_sjf(tasks_processed, current_time, batch_size)
        elif schedule == 'edf':
            current_over_tasks, tasks_processed = task_edf(tasks_processed, current_time, batch_size)
        else:
            raise ValueError("Invalid schedule type")
        sentence = []
        sentence_mid = []
        for task in current_over_tasks:
            sentence_mid.append(task['sentence'])
        sentence.append(sentence_mid)

        start_time = time.time()
        model.predict(sentence)
        end_time = time.time()

        for task in current_over_tasks:
            task_processing_time = end_time - start_time
            task_end_time = current_time + task_processing_time
            task['End_Time'] = task_end_time
            task['time_out'] = 1 if task_end_time > task['true_time'] else 0

        processed_tasks.extend(current_over_tasks)
        current_time = task_end_time

    return processed_tasks


def main():
    tasks = generate_random_tasks(100)
    parser = argparse.ArgumentParser(description='Schedule program')
    parser.add_argument('--glm_path', type=str, default="../model/chatglm2-6b", help='The path of ChatGLM')
    parser.add_argument('--task_batch', type=int, default=16, help='Number of tasks to be sent within a time interval')
    parser.add_argument('--batch_size_predict', type=int, default=16, help='Number of tasks to be predicted at once')
    parser.add_argument('--schedule_mode', type=str, default='sjf', help='Schdule_mode')
    parser.add_argument('--interval_time', type=float, default=0.1, help='Interval time')
    args = parser.parse_args()
    tasks_processed = task_process(tasks, args.interval_time, args.task_batch)
    processed_tasks = glm_predict(GLM_Model(args.glm_path), tasks_processed, args.schedule_mode,
                                  args.batch_size_predict)
    print(processed_tasks)


if __name__ == "__main__":
    main()
