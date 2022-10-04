import gc
import resource

import matplotlib


import matplotlib.pyplot as plt
import pandas as pd

from tp2 import utils
from tp2.perceptron import Perceptron

matplotlib.use("TkAgg")

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
# import gc

def create_animation(data, historic):
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection="3d")
    print(data)

    def animate(i):
        # if not i % 100:
        #     gc.collect()
        ax.clear()
        ax.set_ylim([-2, 8])
        ax.set_xlim([-2, 8])
        ax.set_zlim([-2, 8])

        res = historic[i]

        for j in range(0, len(data)):
            color = 0xFFFFFF * ((res[j][0] + 1) / 2)
            ax.scatter(data[j][0], data[j][1], data[j][2], color="#{:06x}".format(round(color)))

    ani = FuncAnimation(fig, animate, frames=len(historic), interval=1000, repeat=False)
    plt.close()
    ani.save("anim.gif", dpi=300, writer=PillowWriter(fps=1))
    # plt.show()


if __name__ == "__main__":
    perceptron = Perceptron.load()
    # with open('/proc/meminfo', 'r') as mem:
    #     free_memory = 0
    #     for i in mem:
    #         sline = i.split()
    #     if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
    #         free_memory += int(sline[1])
    # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # resource.setrlimit(resource.RLIMIT_AS, (round(free_memory * 1024 * 0.9), hard))
    create_animation(perceptron.data, perceptron.historic)
