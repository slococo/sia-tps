import matplotlib

import matplotlib.pyplot as plt
from tp2.perceptron import Perceptron

matplotlib.use("TkAgg")

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def create_animation(data, historic):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection="3d")
    print(data)

    def animate(i):
        ax.clear()

        ax.set_ylim([-2, 8])
        ax.set_xlim([-2, 8])
        ax.set_zlim([-2, 8])

        res = historic[i]
        for j in range(0, len(data)):
            val = 0xFF * ((res[j][0] + 1) / 2)
            ax.scatter(data[j][0], data[j][1], data[j][2], color="#{:02x}{:02x}{:02x}".format(round(val), round(val), round(val)))

    # ani = FuncAnimation(fig, animate, frames=len(historic), interval=1, repeat=False)
    ani = FuncAnimation(fig, animate, frames=len(historic), interval=90, repeat=False)
    # plt.show()
    plt.close()
    ani.save("anim.gif", writer='PillowWriter', fps=11)
    # plt.show()


if __name__ == "__main__":
    perceptron = Perceptron.load()
    create_animation(perceptron.data, perceptron.historic)
