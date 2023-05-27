import argparse
import crystall_defect_cv as cdcv


def set_cfg(technique: str, **kwargs):
    cdcv.config[technique].update(**kwargs)


def main(inp: str, out: str, ksize: int, threshold: float, power: float,
         technique: str = "sobel_technique"):
    set_cfg(technique, ksize=ksize, threshold=threshold, power=power)

    img = cdcv.open_png(inp)  # "./pictures/006450.png"
    prob_matrix = cdcv.find_defects_probs(img)
    marked_image = cdcv.mark_defects(img, prob_matrix)
    cdcv.save_png(marked_image, out)  # "./output/006460_sad.png"
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CrystalDefect")
    parser.add_argument('inp', help="input file path (.png)", type=str)
    parser.add_argument('out', help="output file path (.png)", type=str)
    parser.add_argument('-k', '--ksize', help="set sobel kernel size", type=int, default=15)
    parser.add_argument('-t', '--threshold', help="set threshold for defect intensity for it to be marked",
                        type=float, default=14.7)
    parser.add_argument('-p', '--power', help="set filter strength for defect intensity", type=float, default=1.7)

    args = parser.parse_args()
    main(**vars(args))
