from model_trainer import train_model

data = {
    'model_choice': 'SelfTalk',
    'dataset': 'vocaset',
    'train_subjects': 'FaceTalk_170728_03272_TA',
    'val_subjects': '',
    'epoch': '1',
    'gpu_choice': 'cuda:0',
}

result = train_model(data)
print("训练返回结果：")
print(result)
