from typing import Literal

class Setting:
    ENCODING_ALGO: Literal["amplitute", "angle"] = "angle"
    DATASET_CHOSEN: Literal["mnist", "fashion_mnist"] = "mnist"
    FINAL_ACTIVATION_FUNCTION: Literal["sigmoid", "custom"] = "custom"
