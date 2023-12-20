from src.models.build_model import build_lstm
from src.utils import load_h5_file
from keras.models import Sequential
import glob

def test_load_h5_file(load_model_path: str):
    model_files = glob.glob(f"{load_model_path}*.h5")
    for file in model_files:
        model = load_h5_file(file)
        assert isinstance(model, Sequential)