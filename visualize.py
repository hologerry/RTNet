import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from cv2 import imread, imwrite
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import trange


visualization_path = 'visualization_RX50_s'

real_image_root_path = '/D_data/Seg/data/object_test/img'

seg_result_folders = ['results_model_RX50',
                      'results_model_RX50_scope1',
                      'results_model_RX50_scope-1',
                      'results_model_RX50_scope+1',
                      'results_model_RX50_fwflow_only',
                      'results_model_RX50_bwflow_only']


labels = ['image', 'default', 'scope1', 'scope-1', 'scope+1', 'fwflow_only', 'bwflow_only']

os.makedirs(visualization_path, exist_ok=True)


for i in trange(2, 4672):
    real_image_path = os.path.join(real_image_root_path, f"out-{i:05d}.jpg")
    seg_img_paths = [os.path.join('results', seg_path, f"out-{i:05d}.jpg") for seg_path in seg_result_folders]
    real_img = [imread(real_image_path)]
    seg_imgs = [imread(seg_img_path) for seg_img_path in seg_img_paths]

    imgs = real_img + seg_imgs

    # plt.axis('off')
    fig = plt.figure(figsize=(8., 8.), dpi=100)
    canvas = FigureCanvasAgg(fig)

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 2),  # creates 2x2 grid of axes
                     axes_pad=0.4,  # pad between axes in inch.
                     aspect=True,
                    )

    for ax, im, l in zip(grid, imgs, labels):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(l)
        ax.axis('off')

    plt.tight_layout()
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    out_path = os.path.join(visualization_path, f"out-{i:05d}.jpg")

    imwrite(out_path, img)
    plt.close()

