import json
import importlib.resources


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
