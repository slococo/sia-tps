import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from graph import plot
from tp2.ej2.wrapper import Wrapper

matplotlib.use("TkAgg")

from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


def create_animation(data, historic):
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection="3d")

    def animate(i):
        ax.clear()
        ax.set_title(f"iter: {i}")
        res = np.squeeze(historic[i])
        plot(data[:, 1:], res, ax)

    ani = FuncAnimation(fig, animate, frames=len(historic), interval=20, repeat=False)
    plt.close()
    ani.save("anim.gif", fps=200)

    # ani.save("anim.mp4", writer=FFMpegWriter(fps=60))


if __name__ == "__main__":
    wrapper = Wrapper.load()
    if wrapper.historic:
        create_animation(wrapper.data, wrapper.historic)
