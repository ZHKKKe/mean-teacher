import os
import re
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def read_log(log_name):
    assert os.path.exists(log_name) and os.path.isfile(log_name) 

    content = None
    with open(log_name) as log:
        content = log.read()
    return content

def read_logs(log_path):
    assert os.path.exists(log_path) and os.path.isdir(log_path)

    names = []
    contents = {}
    for log_name in os.listdir(log_path):
        log_file = os.path.join(log_path, log_name)
        names.append(log_name)
        contents[log_name] = read_log(log_file)

    return contents, names

def parse_info(logs, names, pattern):
    datas = {}
    for name in names:
        info = re.findall(pattern, logs[name])
        if name.startswith('fmcb'):
            data = [float(_.split(' ')[-1]) for idx, _ in enumerate(info) if idx % 4 == 3]
        elif name.startswith('mt'):
            data = [float(_.split(' ')[-1]) for idx, _ in enumerate(info) if idx % 2 == 1]
        
        datas[name] = data
    return datas



def draw_curves(name, labels, datas):
    plt.title(name)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    for label in labels:
        y = datas[label]
        x = [_ for _ in range(0, len(y))]
        plt.plot(x, y, label=label)
    
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    LOG_PATH = './logs'
    PATTERN = r'\* Prec@1 \d+\.\d+'
    
    logs, names = read_logs(LOG_PATH)
    datas = parse_info(logs, names, PATTERN)    
    draw_curves('test', names, datas)