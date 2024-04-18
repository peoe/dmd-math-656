import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from dmd_segmentation import StreamingDMD, Image
from logging import basicConfig, INFO
from matplotlib.animation import ArtistAnimation, PillowWriter
from tqdm import tqdm


START_FRAME = 250 # the data really only starts at frame no 250


def main():
    basicConfig(level=INFO)

    im = Image()
    frames = []
    for index, frame in enumerate(im.get_images("pedestrian_detection_dataset/pedestrians/input")):
        if index >= START_FRAME:
            f = np.array(frame.ravel(), dtype=np.float64)
            nf = np.linalg.norm(f)
            frames.append(f / nf)

    X = frames[:-1]
    Y = frames[1:]
    x_len = len(X)

    data_iter = lambda: zip(X, Y)  # noqa: E731
    hist_lens = [5]
    rrank = 10
    predictions = []
    fgs = []
    for hist_len in hist_lens:
        m_str = StreamingDMD(max_rank=rrank, max_hist=hist_len)
        str_errs = 1e-16 * np.ones(x_len - 1)
        min_res = 1e-16 * np.ones(x_len - 1)
        for iter, (x, y) in tqdm(enumerate(data_iter()), total=x_len, desc=f"Streaming QR Memory size {hist_len:2}", ncols=90):
            if iter > 0:
                ny = np.linalg.norm(y)
                pred = m_str(x)
                predictions.append(pred)
                str_errs[iter - 1] = np.linalg.norm(y - pred) / ny

                # identify background modes
                res, modes, amps = m_str.residuals
                logamps = np.log(amps)
                min_res[iter - 1] = np.min(np.abs(logamps))
                tol = 1e-3
                bg_indices = list(np.where(np.abs(logamps) < tol)[0])
                if not bg_indices:
                    bg_indices = [0]
                bg = m_str.reconstruct(modes[:, bg_indices], amps[bg_indices], [hist_len]).real.reshape(-1)
                bg /= np.linalg.norm(bg)
                fg = np.abs(x - bg)
                fgs.append(fg)

            m_str.update(x, y)

        print(f"QR Prediction Error Metrics with Memory {hist_len}: max - {np.max(str_errs):6.3e} min - {np.min(str_errs):6.3e} mean - {np.mean(str_errs):6.3e} median - {np.median(str_errs):6.3e}")
        plt.plot(range(len(str_errs)), str_errs, label=f"Streaming QR with mem {hist_len}")
        plt.semilogy(min_res, linestyle="none", marker="x")

    plt.legend(np.repeat(hist_lens, 2))

    plt.show(block=False)

    fig, ax = plt.subplots()
    ims = []
    for iter, fg in enumerate(fgs):
        mat = fg.reshape(im.resolution)
        mat = np.where(mat > 5e-4, 1, 0)
        mplot = ax.imshow(mat, cmap=mpl.colormaps["Greys_r"], animated=True)
        ax.set_title("Foreground Pedestrian")
        title = ax.text(0, -5, f"DMD Frame {iter}")
        if iter == 0:
            ax.imshow(mat, cmap=mpl.colormaps["Greys_r"])
            ax.set_title("Foreground Pedestrian")
        ims.append([mplot, title])
    ani = ArtistAnimation(fig, ims, interval=300)
    ani.save("pedestrian_foreground.gif", writer=PillowWriter(fps=30))


if __name__ == "__main__":
    main()
