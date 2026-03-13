import os
from mltk.utils.path import create_tempdir

# This contains the path to the pre-trained model in ONNX model format
# For this tutorial, we use the one downloaded from above
# Update this path to point to your specific model if necessary
ONNX_MODEL_PATH = f'{cifar10_matlab_model_example_dir}/cifar10_matlab_model.onnx'

# This contains the path to our working directory where all
# generated, intermediate files will be stored.
# For this tutorial, we use a temp directory.
# Update as necessary for your setup
WORKING_DIR = create_tempdir('cifar10_matlab_model_onnx_to_tflite')



assert os.path.exists(ONNX_MODEL_PATH), f'The provided ONNX_MODEL_PATH does not exist at: {ONNX_MODEL_PATH}'
os.makedirs(WORKING_DIR, exist_ok=True)


# Use the filename for the model's name
MODEL_NAME = os.path.basename(ONNX_MODEL_PATH)[:-len('.onnx')]


print(f'ONNX_MODEL_PATH = {ONNX_MODEL_PATH}')
print(f'MODEL_NAME = {MODEL_NAME}')