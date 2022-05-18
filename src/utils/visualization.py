import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def save_sample_output(image, bboxes, mask, fig_name):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(image)
    ax2.imshow(image)
    for bbox in bboxes:
        if max(bbox) > 512 or min(bbox) < 0:
            continue
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="red",
            facecolor=None,
            fill=False
        )
        ax2.add_patch(rect)

    ax3.imshow(mask)
    plt.savefig(fig_name)