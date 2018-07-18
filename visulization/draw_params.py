import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# max_lr = base_lr * ngpu
def calculate_cos_lr(epochs, steps, max_lr, lr_rampdown_epochs=None, lr_rampup=0, init_lr=0, cos_scale=0.5):
    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length

    def cosine_rampdown(current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        assert 0 <= current <= rampdown_length
        return float(cos_scale * (np.cos(np.pi * current / rampdown_length) + 1))

    key = []
    value = []
    for epoch in range(0, epochs):
        for step in range(0, steps):
            lr = max_lr
            global_step = epoch + step / steps

            # rampup
            scale = linear_rampup(global_step, lr_rampup)
            lr = scale * (max_lr - init_lr) + init_lr

            # cos decline
            if lr_rampdown_epochs is not None:
                assert lr_rampdown_epochs >= epochs
                lr *= cosine_rampdown(global_step, lr_rampdown_epochs)

            key.append(global_step)
            value.append(lr)
    data = [key, value]
    return data

def sigmoid_up(total_steps):
    import math
    key = []
    value = []
    for step in range(0, total_steps):
        x = 1.0 - max(0.0, step) / total_steps
        x = math.exp(-5.0 * x * x)
        key.append(step)
        value.append(x)
    data = [key, value]
    return data

def line_up(total_steps):
    import math
    key = []
    value = []
    for step in range(0, total_steps):
        x = max(0.0, step) / total_steps
        # x = math.exp(-5.0 * x * x)
        key.append(step)
        value.append(x)
    data = [key, value]
    return data


def draw(name, legend, data):
    assert len(legend) == len(data)

    plt.title(name)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for idx in range(0, len(legend)):
        plt.plot(data[idx][0], data[idx][1], label=legend[idx])

    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    # configs = [
    #     {'legend': 'mt4000', 'epochs': 300, 'steps': 474, 'max_lr': 0.05 * 1,
    #     'lr_rampdown_epochs': 350, 'lr_rampup': 0, 'init_lr': 0, 'cos_scale': 0.5},
    # ]


    # data = []
    # legend = []
    # for config in configs:
    #     data.append(calculate_cos_lr(config['epochs'], config['steps'], config['max_lr'],
    #                 config['lr_rampdown_epochs'], config['lr_rampup'], config['init_lr'], config['cos_scale']))
    #     legend.append(config['legend'])

    # draw('lr_draw', legend, data)

    configs = [
        {'legend': 'sigmoid_up', 'total_steps': 40000},
        {'legend': 'line_up', 'total_steps': 40000},
    ]

    data = []
    legend = []
    for config in configs:
        if config['legend'] == 'sigmoid_up':
            data.append(sigmoid_up(config['total_steps']))
        elif config['legend'] == 'line_up':
            data.append(line_up(config['total_steps']))
        legend.append(config['legend'])
    # print(data)
    # print(legend)
    draw('draw', legend, data)
