import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

def plot_loss_train_from_log_file(log_file="extra/train.log",out_file="loss.png"):
    losses=[]
    with open(log_file,"r") as f:
        for line in f:
            if "mean train loss" in line.strip():
                losses.append(float(line.strip().split(":")[-1]))

    data={"steps":[f"{i*5}k" for i in range(len(losses))],"losses":losses,"losses2":[math.sqrt(i) for i in losses]}
    fig, ax = plt.subplots(1, 1)
    ax.plot('steps', 'losses', data=data, marker='o', markerfacecolor='blue', markersize=3, color='skyblue',linewidth=4)
    ax.set_title('Training loss')
    ax.text(35, 1.2, "loss drops fast at the beginning", size=10, rotation=0.,
             ha="center", va="center",
             )
    ax.text(90, 0.7, "loss remains flat later", size=10, rotation=0.,
             ha="center", va="center",
             )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file)

    plt.show()

if __name__ == '__main__':
    plot_loss_train_from_log_file()