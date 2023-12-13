import torch
from models.cofima import CoFiMA
from models.seqfinetune import SeqFinetune
from models.simplecil import SimpleCIL

def get_model(model_name, args):
    name = model_name.lower()
    if 'cofima' in name:
        return CoFiMA(args)
    elif 'seqfinetune' in name:
        return SeqFinetune(args)
    elif 'simplecil' in name:
        return SimpleCIL(args)
    else:
        assert 0
