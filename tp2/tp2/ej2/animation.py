import matplotlib
import matplotlib.pyplot as plt
from tp2.ej2.wrapper import Wrapper

matplotlib.use("TkAgg")

from matplotlib.animation import FuncAnimation, PillowWriter


def create_animation(data, historic):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection="3d")

    def animate(i):
        ax.clear()

        ax.set_ylim([-2, 8])
        ax.set_xlim([-2, 8])
        ax.set_zlim([-2, 8])

        res = historic[i]
        for j in range(0, len(data)):
            val = 0xCC * ((res[j][0] + 1) / 2)
            ax.scatter(
                data[j][0],
                data[j][1],
                data[j][2],
                color="#{:02x}{:02x}{:02x}".format(round(val), 0x00, round(val / 3)),
            )

    ani = FuncAnimation(fig, animate, frames=len(historic), interval=86, repeat=False)
    plt.close()
    ani.save("anim.gif", writer="PillowWriter", fps=12)


if __name__ == "__main__":
    wrapper = Wrapper.load()
    if wrapper.historic:
        create_animation(wrapper.data, wrapper.historic)
