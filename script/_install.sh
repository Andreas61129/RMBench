echo "Installing a matching CUDA toolkit (nvcc) into the conda env ..."
# Needed so building CUDA extensions (pytorch3d, curobo) uses a compiler whose
# major version matches torch's CUDA build (cu128), regardless of what CUDA
# version is installed system-wide (e.g. a newer system CUDA on rolling-release
# distros like CachyOS/Arch will otherwise be picked up first and fail the
# version check in torch.utils.cpp_extension). Also pulls in gcc_linux-64/
# gxx_linux-64 (a host compiler CUDA 12.8 actually supports, vs. whatever
# newer system gcc a rolling-release distro ships).
conda install -c nvidia -c conda-forge cuda-toolkit=12.8 gcc_linux-64 gxx_linux-64 -y
export CUDA_HOME=$CONDA_PREFIX
# The conda cuda-toolkit package splits headers into targets/<arch>/include
# instead of $CUDA_HOME/include; CUDA_INC_PATH is explicitly checked by
# torch.utils.cpp_extension as a fallback include dir.
export CUDA_INC_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
# Point the host compiler at the conda-provided gcc/g++ instead of whatever
# CC/CXX happen to be in the current shell (gcc_linux-64's activation hook
# only fires on `conda activate`, not `conda install`, so it won't have taken
# effect yet in this script even though the packages are now present).
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

echo "Installing the necessary packages ..."
pip install -r script/requirements.txt

echo "Installing pytorch3d ..."
# cd third_party/pytorch3d_simplified
# pip install -e .
# cd ../..
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

echo "Installing Curobo ..."
cd envs
if [ ! -d curobo ]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
fi
cd curobo
rm -rf build
pip install -e . --no-build-isolation
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download asserts from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
