import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from cv2 import imread, imwrite
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import trange


visualization_path = 'visualization_different'

real_image_root_path = '/D_data/Seg/data/object_test/img'

seg_result_folders = ['results/results_model_R34',
                      'results/results_model_R34_fwflow_only',
                      'results/results_model_RX50',
                      'results/results_model_RX50_fwflow_only',
                      '../RTNet_output/pseg_scope40_r34_gen_mobilenet_small_bs2/results',
                      '../RTNet_output/pseg_scope40_r34_gen_mobilenet_small_bs2_fw_only/results',
                      '../RTNet_output/pseg_scope40_rx50_gen_mobilenet_small_bs2/results',
                      '../RTNet_output/pseg_scope40_rx50_gen_mobilenet_small_bs2_fw_only/results']


labels = ['image', 'image', 'pretrained_R34', 'pretrained_R34_fw', 'pretrained_RX50', 'pretrained_RX50_fw', 'our_R34', 'our_R34_fw', 'our_RX50', 'our_RX50_fw']

os.makedirs(visualization_path, exist_ok=True)


for i in trange(2, 4672):
    real_image_path = os.path.join(real_image_root_path, f"out-{i:05d}.jpg")
    seg_img_paths = [os.path.join(seg_path, f"out-{i:05d}.jpg") for seg_path in seg_result_folders]
    real_img = [imread(real_image_path), imread(real_image_path)]
    seg_imgs = [imread(seg_img_path) for seg_img_path in seg_img_paths]

    imgs = real_img + seg_imgs

    # plt.axis('off')
    fig = plt.figure(figsize=(10., 10.), dpi=100)
    canvas = FigureCanvasAgg(fig)

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 2),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     aspect=True,
                    )

    for im_idx, (ax, im, l) in enumerate(zip(grid, imgs, labels)):
        # Iterating over the grid returns the Axes.
        if im_idx == 1:
            continue
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

