import crystall_defect_cv as cdcv
from crystall_defect_cv.processing_modules import module_sobel

image = cdcv.open_png("test.png")

module_sobel.sobel_technique(image)

# defects_matrix = cdcv.find_defects_probs(image)

# defects_image = cdcv.mark_defects(image, defects_matrix, min_confidence_threshold=0.3)

# cdcv.save_png(defects_image, "test_processed.png")


