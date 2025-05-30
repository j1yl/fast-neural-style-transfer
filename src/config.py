CONTENT_WEIGHT = 1e3
STYLE_WEIGHT = 1e8
COLOR_WEIGHT = 10.0

STYLE_LAYER_WEIGHTS = [1.0, 1.0, 2.0, 4.0]  # relu1_2, relu2_2, relu3_3, relu4_3
STYLE_LOSS_SCALING = 1  # Set to 1.0 for clarity; adjust if needed

BRIGHTNESS_WEIGHT = 0.5
CONTRAST_WEIGHT = 0.3
COLOR_MATCHING_WEIGHT = 5.0
