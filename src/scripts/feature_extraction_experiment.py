import torch
import optuna
from src.models import get_EfficientNet_B3, get_EfficientNet_V2_S, get_EfficientNet_V2_M, get_EfficientNet_V2_M, \
    get_MobileNet_V3_L, get_Swin_V2_S, get_Swin_V2_B 

# biggest models that can be ran: EN_B3, EF_V2_M?, MN_V2_L, S_V2_S
models = [get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L, get_Swin_V2_S]

def objective(trial):
    # DATA AUGMENT PARAMETERS
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [8, 16, 32, 64, 128])
    LR = trial.suggest_float('LR', 0.0001, 0.01)
    WEIGHT_DECAY = trial.suggest_float('WEIGHT_DECAY', 0.0001, 0.005)

    # PREPARE DATA
    transforms = make_transforms_rgb(INPUT_DIM, V_FLIP, H_FLIP, PERSPECTIVE, PERSPECTIVE_P, ROTATION, BRIGHTNESS, CONTRAST, SATURATION, HUE)
    datasets = make_datasets(transforms, './cards/train', './cards/valid', './cards/test')
    dataloaders = make_dataloaders(datasets, BATCH_SIZE)

    # INSTANTIATING OUR MODEL, LOSS AND OPTIMIZER
    model = MiniVGG_RGB(INPUT_DIM, len(datasets['train'].classes), DROPOUT).to(device)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LR) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # TRAINING AND EVALUATING THE MODEL
    train_model_with_scheduler(device, model, dataloaders, EPOCHS, criterion, optimizer, scheduler)
    _, y_true, y_pred, _ = evaluate_model_tta(device, model, dataloaders['test'])
    return f1_score(y_true, y_pred, average='weighted')


study = optuna.create_study(study_name='vgg_rgb_25_epochs_112_inputsize_3', storage='sqlite:///cnn_classifier.db', direction="maximize")
study.optimize(objective, n_trials=50)
