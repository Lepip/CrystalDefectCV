import json
import importlib.resources

"""
Sobel_technique:
ksize: size of the kernel.
threshold: threshold of the defect intensity.
blur_size: size of the bluring kernel of the possible defect pixels.
power: threshold, but after blur and exponential.
blur_threshold: how much value has to tip up from the mean to be considered a defect.

Scharr_technique:
The same, as for sobel_technique with ksize=3.

use_sobel:
if true, enables sobel_technique to be used.

use_scharr:
if true, enables scharr_technique to be used.
"""
standard_config_json = """
{
  "sobel_technique": {
    "ksize": 15,
    "threshold": 14.7,
    "blur_size": 37,
    "power": 1.7,
    "blur_threshold": 20
  },

  "scharr_technique": {
    "threshold": 4,
    "blur_size": 14,
    "power": 1,
    "blur_threshold": 15
  },

  "use_sobel": true,
  "use_scharr": false
}
"""
config = json.loads(standard_config_json)
