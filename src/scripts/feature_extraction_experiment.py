import torch
import optuna
from src.models import get_EfficientNet_B3, get_EfficientNet_V2_S, get_EfficientNet_V2_M, get_EfficientNet_V2_M, \
    get_MobileNet_V3_L, get_Swin_V2_S, get_Swin_V2_B 

# biggest models that can be ran: EN_B3, EF_V2_M?, MN_V2_L, S_V2_S
models = [(get_EfficientNet_B3, ), (get_EfficientNet_V2_S), (get_MobileNet_V3_L), (get_Swin_V2_S)]

def objective(trial):
    # DATA AUGMENT PARAMETERS
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [8, 16, 32, 64, 128])
    LR = trial.suggest_float('LR', 0.0001, 0.01)
    WEIGHT_DECAY = trial.suggest_float('WEIGHT_DECAY', 0.0001, 0.005)

    # PREPARE DATA

    # INSTANTIATING OUR MODEL, LOSS AND OPTIMIZER

    # MAXIMIZE VAL ACC
    return 


study = optuna.create_study(study_name='vgg_rgb_25_epochs_112_inputsize_3', storage='sqlite:///model_selection.db', direction="maximize")
study.optimize(objective, n_trials=50)
