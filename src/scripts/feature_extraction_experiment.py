import torch
import optuna
from src.models import get_EfficientNet_B3, get_EfficientNet_V2_S, get_EfficientNet_V2_M, get_EfficientNet_V2_M, \
    get_MobileNet_V3_L, get_Swin_V2_S, get_Swin_V2_B 

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L, get_Swin_V2_S]
models_max_bs = [32, 32, 128, 256,]

def objective(trial):
    # DATA AUGMENT PARAMETERS
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [8, 16, 32, 64, 128])

    LR = trial.suggest_float('LR', 0.0001, 0.01)
    WEIGHT_DECAY = trial.suggest_float('WEIGHT_DECAY', 0.0001, 0.005)

    MODEL_IDX = trial.suggest_inte('MODEL_IDX', 0, 3)

    model, transforms = models[MODEL_IDX]()
    model = model.to(device)

    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)

    print(train_df[TARGETS].describe()) 
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

    train_ds = MMFood100KDataset(train_df, transform=transforms, input='resized_img_path')
    test_ds = MMFood100KDataset(test_df, transform=transforms, input='resized_img_path')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    del df, train_df, test_df, train_ds, test_ds 

    losses = train_eval_loop(
        model = model,
        epochs = EPOCHS,
        train_loader = train_loader,
        test_loader = test_loader,
        criterion = torch.nn.L1Loss(), # MAE
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01),
        device = DEVICE
    )

    # PREPARE DATA

    # INSTANTIATING OUR MODEL, LOSS AND OPTIMIZER

    # MAXIMIZE VAL ACC
    return 


study = optuna.create_study(study_name='vgg_rgb_25_epochs_112_inputsize_3', storage='sqlite:///model_selection.db', direction="maximize")
study.optimize(objective, n_trials=50)
