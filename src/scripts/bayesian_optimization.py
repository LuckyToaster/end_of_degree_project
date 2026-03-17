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
    TODO

'''
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
)

study.optimize(objective, n_trials=100)   
'''

print_model_sizes()
