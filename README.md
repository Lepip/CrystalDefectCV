# Crystal defect CV

A library to mark defects on images of crystals.

## One click usage

### Raw Python

Install requiremenst.txt:

```commandline
pip install -r requirements.txt
```

Usage:

```commandline
python main.py [-h] [-k KSIZE] [-t THRESHOLD] [-p POWER] inp out
```

Arguments:
- `[-k KSIZE]` `(standard=15)` -- size of the kernel used. When increased, decreases sensitivity to smaller defects. Can only be odd.
- `[-t THRESHOLD]` `(standard=14.7)` -- threshold of the defect intensity to be considered a defect. Basically larger threshold equals smaller marked areas.
- `[-p POWER]` `(standard=1.7)` -- basically the same as threshold, but exponential.
- `inp` -- path to the png image to mark.
- `out` -- path to save the output png to.
 

### As executable

You can download compiled executable for Windows or Linux here: https://disk.yandex.ru/d/KMenXUanGjCQrw

#### Using Pyinstaller

Install PyInstaller:

```commandline
pip install -U pyinstaller
```

Compile the code into a single-file executable by yourself:

```commandline
pyinstaller main.py -F
```

## Library usage

First off, import the library:

```python
import crystal_defect_cv as cdcv
```

Basic usage:

```python
png = cdcv.open_png(DIRECTORY)
defect_matrix = cdcv.find_defects_probs(png)
marked_png = cdcv.mark_defects(png, defect_matrix)
cdcv.save_png(marked_png, SAVE_DIRECTORY)
```

All library contents:
- `open_png(img_path)` -- opens the image.
- `find_defects_probs(png)` -- creates a matrix of where the algorithm thinks the defects are at.
- `mark_defects(png, defect_matrix, lighten_up=True)` -- marks all nonzero pixels in defect_matrix on the lightened up png.
- `save_png(png, save_path)` -- saves the png to save_dir.
- `modules_dict` -- dictionary, that has the processing modules with their name as key. 
Every processing module is a function similar to `find_defects`.
- `config` -- json-based dictionary of configurations. Described in `crystal_defect_cv/processing_modules/config.py`. Can be changed as a dictionary to configure modules.
- `mark_all_in_directory(operator, directory, save_directory)` -- uses `operator` 
(like `modules_dict["sobel_technique"]`) on all the png images in the `directory` and saves marked images in `output_directory`.
