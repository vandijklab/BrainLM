module load miniconda #only needed if you're on McCleary
conda create --name brainlm python=3.10.13
module load miniconda
conda activate brainlm


pip install packaging
pip install ninja
conda install -c conda-forge cudatoolkit-dev -y
pip install torch torchvision torchaudio
pip install flash_attn --no-build-isolation
pip install -r requirements.txt --no-build-isolation