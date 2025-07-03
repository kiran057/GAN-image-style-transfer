from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
import os
import uuid

# Use only define_G for all models trained using pytorch-CycleGAN-and-pix2pix
from models import networks
from models.generator_model import Generator

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define which models use batch norm (Pix2Pix) and which use instance norm (CycleGAN)
PIX2PIX_MODELS = ['pix2pix_day2night']
GENERATOR_MODEL_MODELS = ['summer2winter', 'winter2summer']

MODEL_PATHS = {
    # Pix2Pix checkpoints
    'pix2pix_day2night': 'models/latest_net_G.pth',
   
    'summer2winter': 'models/gen_S2W_checkpoint_clean.pth',
    'winter2summer': 'models/gen_W2S_checkpoint_clean.pth',
    # Optional CycleGAN models
}

# Preprocessing (matches training 'resize_and_crop')
transform = transforms.Compose([
    transforms.Resize(286, Image.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

# Inverse transform
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# Function to load the correct model
_cached_models = {}
def get_model(name):
    if name in _cached_models:
        return _cached_models[name]

    path = MODEL_PATHS.get(name)
    if path is None or not os.path.exists(path):
        raise ValueError(f"Unknown or missing model: {name}")

    if name in GENERATOR_MODEL_MODELS:
        model = Generator(img_channels=3).to(device)
    else:
        norm_type = 'batch' if name in PIX2PIX_MODELS else 'instance'
        netG_type = 'unet_256' if name in PIX2PIX_MODELS else 'resnet_9blocks'
        model = networks.define_G(
            input_nc=3, output_nc=3, ngf=64,
            netG=netG_type, norm=norm_type,
            use_dropout=False, init_type='normal', init_gain=0.02,
            gpu_ids=[]
        ).to(device)

    state_dict = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model '{name}': {e}")

    model.eval()
    _cached_models[name] = model
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Save uploaded file
    file = request.files.get('image')
    style = request.form.get('model')
    if file is None or style is None:
        return "Missing image or model selection", 400

    uid = str(uuid.uuid4())
    infile = os.path.join(UPLOAD_FOLDER, uid + '.png')
    file.save(infile)

    # Prepare input
    img = Image.open(infile).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    model = get_model(style)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Process output
    out_img = inv_transform(output_tensor.squeeze(0).cpu())
    outfile = os.path.join(RESULT_FOLDER, uid + '.png')
    out_img.save(outfile)

    return render_template('result.html', input_image=infile, output_image=outfile)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
