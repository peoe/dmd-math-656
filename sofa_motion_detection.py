import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser
from dmd_segmentation import StreamingDMD, Image
from logging import basicConfig, INFO
from matplotlib.animation import ArtistAnimation, PillowWriter
from tqdm import tqdm


START_FRAME = 50 # the data really only starts at frame no. 250


def main():
    parser = ArgumentParser(
        prog='SofaDetection',
        description='Detect the movement of sofa dateset using Streaming DMD.'
    )
    parser.add_argument("--output", choices=["all", "gifs", "plots", "matrices", "none"], default="none")
    output = parser.parse_args().output
    basicConfig(level=INFO)

    im = Image()
    frames = []
    for index, frame in enumerate(im.get_images(r"data/pedestrian detection dataset/sofa/input")):
        if index >= START_FRAME:
            f = np.array(frame.ravel(), dtype=np.float64)
            nf = np.linalg.norm(f)
            frames.append(f / nf)

    X = frames[:-1]
    Y = frames[1:]
    x_len = len(X)

    data_iter = lambda: zip(X, Y)  # noqa: E731
    hist_lens = [5, 10, 15, 20]
    max_rank = 5
    results = {
        "min_ress": 1e-16 * np.ones((len(hist_lens), x_len - 1)),
        "fgs": {},
    }
    for index, hist_len in enumerate(hist_lens):
        m_str = StreamingDMD(max_rank=max_rank, max_hist=hist_len)
        str_ress = 1e-16 * np.ones(x_len - 1)
        fgs = []
        for iter, (x, y) in tqdm(enumerate(data_iter()), total=x_len, desc=f"Streaming QR Memory size {hist_len:2}", ncols=90):
            if iter > 0:
                # identify background modes
                res, modes, amps = m_str.residuals
                logamps = np.log(amps.astype(np.cdouble)) # this avoids nan problems
                str_ress[iter - 1] = np.min(res)
                bg_indices = list(np.where(np.abs(logamps) < 1e-3)[0])
                if not bg_indices:
                    bg_indices = [0]

                # background model
                bg = m_str.reconstruct(m_str._Q @ modes[:, bg_indices], amps[bg_indices], [hist_len]).real.reshape(-1)
                bg /= np.linalg.norm(bg)

                # foreground
                fg = np.abs(x - bg)
                fgs.append(fg)

            m_str.update(x, y)

        results["min_ress"][index, :] = str_ress
        results["fgs"][hist_len] = fgs

    if output == "matrices" or output == "all":
        ress = pd.DataFrame(results["min_ress"].T)
        ress.to_csv("sofa_motion_res.csv", header=hist_lens, index_label="index")

    if output == "gifs" or output == "all":
        for hist_len in hist_lens:
            fig, ax = plt.subplots()
            ims = []
            for iter, fg in enumerate(fgs):
                mat = fg.reshape(im.resolution)
                mat = np.where(mat > 4e-4, 1, 0)
                mplot = ax.imshow(mat, cmap=mpl.colormaps["Greys_r"], animated=True)
                ax.set_title("Motion Sofa")
                title = ax.text(0, -5, f"DMD Frame {iter}")
                if iter == 0:
                    ax.imshow(mat, cmap=mpl.colormaps["Greys_r"])
                    ax.set_title("Motion Sofa")
                ims.append([mplot, title])
            ani = ArtistAnimation(fig, ims, interval=300)
            ani.save(f"sofa_motion_hist_{hist_len}.gif", writer=PillowWriter(fps=30))

    if output == "plots" or output == "all":
        plt.figure()
        plt.semilogy(results["min_ress"].T)
        plt.title(f"DMD Residuals for Sofa Motion Detection with Memory Size {hist_len}")
        plt.xlabel("Frame")
        plt.ylabel("Residual")
        plt.legend(hist_lens)
        plt.show()


if __name__ == "__main__":
    main()
