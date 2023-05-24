import json
import importlib.resources

standard_config = importlib.resources.open_text("crystall_defect_cv.processing_modules", "standard_config.json")
config = json.load(standard_config)
