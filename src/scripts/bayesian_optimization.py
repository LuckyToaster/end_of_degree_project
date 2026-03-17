from torchinfo import summary
import optuna
from src.models import get_EfficientNet_B3, \
        get_EfficientNet_V2_S, get_EfficientNet_V2_M, \
        get_MobileNet_V3_L, \
        get_ShuffleNet_V2_X2_0, \
        get_Swin_V2_T, get_Swin_V2_S, get_Swin_V2_B

models_dict = [ 
    { 'name': 'EffientNet_B3', 'func': get_EfficientNet_B3 },
    { 'name': 'EfficientNet_V2_S', 'func': get_EfficientNet_V2_S },
    { 'name': 'EffientNet_V2_M', 'func': get_EfficientNet_V2_M },
    { 'name': 'MobileNet_V3_L', 'func': get_MobileNet_V3_L },
    #{ 'name': 'ShuffleNet_V2_X2_0', 'func': get_ShuffleNet_V2_X2_0 },
    { 'name': 'Swin_V2_T', 'func': get_Swin_V2_T },
    { 'name': 'Swin_V2_S', 'func': get_Swin_V2_S },
    { 'name': 'Swin_V2_B', 'func': get_Swin_V2_B }
]


def print_model_sizes():
    for obj in models_dict:
        model, _ = obj['func'](verbose=False)
        print(f'{obj['name']}: {vram_dict(model)}')


def vram_dict(model, img_dims=(3,224,224)):
    '''
    '''
    channels, height, width = img_dims
    stats = summary(model, input_size=(1, channels, height, width), verbose=0)
    # used in VRAM formula
    params = stats.total_param_bytes / (1024 ** 2)
    input = stats.total_input / (1024 ** 2)
    forward = stats.total_output_bytes / (1024 ** 2)
    # VRAM = (params_size * 4) + [(forward_size + input) * batch_size]
    vram_gb = lambda bs: ((params * 4) + (forward + input) * bs) / 1024
    return list(map(lambda bs: (bs, round(vram_gb(bs), 3)), [32, 64, 128, 256]))
    # return list(map(lambda bs: {'batch_size': bs, 'gb_vram': round(vram_gb(bs), 3)}, [32, 64, 128, 256]))
    

def objective(trial):
    # DATA AUGMENT PARAMETERS
    V_FLIP = trial.suggest_float('V_FLIP', 0, 0.3)
    H_FLIP = trial.suggest_float('H_FLIP', 0, 0.3)
    PERSPECTIVE = trial.suggest_float('PERSPECTIVE', 0, 0.3)
    PERSPECTIVE_P = trial.suggest_float('PERSPECTIVE_P', 0, 0.3)
    BRIGHTNESS = trial.suggest_float("BRIGHTNESS", 0, 0.5)
    CONTRAST = trial.suggest_float("CONTRAST", 0, 0.5)
    ROTATION = trial.suggest_int("ROTATION", 0, 20)
    # MODEL PARAMETERS
    DROPOUT = trial.suggest_float('DROPOUT', 0.2, 0.5)
    # HYPERPARAMETERS
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [8, 16, 32, 64, 128, 256])
    LR = trial.suggest_float('LR', 0.0001, 0.01)
    WEIGHT_DECAY = trial.suggest_float('WEIGHT_DECAY', 0.0001, 0.005)

    datasets = make_datasets(transforms, './cards/train', './cards/valid', './cards/test')
    dataloaders = make_dataloaders(datasets, BATCH_SIZE)

    # INSTANTIATING OUR MODEL, LOSS AND OPTIMIZER
    model = MiniVGG_RGB(INPUT_DIM, len(datasets['train'].classes), DROPOUT).to(device)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # TRAINING AND EVALUATING THE MODEL
    train_model_with_scheduler(device, model, dataloaders, EPOCHS, criterion, optimizer, scheduler)
    _, y_true, y_pred = evaluate_model(device, model, dataloaders['test'])
    return f1_score(y_true, y_pred, average='weighted')


'''
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
)

study.optimize(objective, n_trials=100)   
'''

print_model_sizes()
