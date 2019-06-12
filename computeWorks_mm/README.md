# computeWorks_examples
Matrix multiplication example performed with OpenMP, OpenACC, BLAS, cuBLAS, and CUDA

## Getting Started
This example requires the following packages:
- CUDA Toolkit 10.1
- PGI CE Compiler 19.4

Optional:
- Eclipse IDE C/C++
- Docker CE + NVIDIA-Docker v2
  - PGI Docker image

**The following installation instructions have been tested on Ubuntu 18.04 and CUDA 10.0+.**

**OpenACC profiling with NVIDIA driver 418.67 and above requires elevated permissions. See [here](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti).** 

**You can achieve this one of two ways.**

1. **Run command with sudo**
```bash
sudo LD_LIBRARY_PATH=/usr/local/cuda/extra/CUPTI/lib64:$LD_LIBRARY_PATH ./computeWorks_mm
```
2. **Following [Administration instructions](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters#SolnAdminTag).**
```bash
sudo systemctl isolate multi-user # Stop the window manager
modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia # Unload dependent modules
cd /etc/modprobe.d/
sudo touch nvidia.conf # Create file named nvidia.conf
sudo su # Switch to root
sudo echo -e "options nvidia "NVreg_RestrictProfilingToAdminUsers=0"" > nvidia.conf
sudo reboot
```

## Installation
### CUDA -> [more details](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu-x86_64-deb)
1. Download [CUDA Toolkit](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)
2. Install (assuming file is in *~/Downloads*)
```bash 
sudo dpkg -i ~/Downloads/cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```
3. Add paths to *~/.bashrc*
```bash
echo -e "\n# CUDA paths" >> ~/.bashrc
echo -e "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
```
- The CUPTI directory is required for OpenACC profiling

### PGI Community Edition (Bare Metal) -> [more details](https://www.pgroup.com/resources/docs/19.4/x86/pgi-install-guide/index.htm#install-linux-pgi)

**Skip this step if you prefer to run utilize the PGI compiler in a Docker container.**

1. Download [PGI CE Compiler](https://www.pgroup.com/support/download_community.php?file=pgi-community-linux-x64)
2. Install (assuming file is in *~/Downloads*)
```bash
export PGI_SILENT=true
export PGI_ACCEPT_EULA=accept
export PGI_INSTALL_DIR=/opt/pgi
export PGI_INSTALL_TYPE=single
export PGI_INSTALL_NVIDIA=true
export PGI_INSTALL_JAVA=true
export PGI_INSTALL_MPI=false
export PGI_MPI_GPU_SUPPORT=false
mkdir -p ~/Downloads/tmp
tar xpfz ~/Downloads/pgilinux-2019-194-x86-64.tar.gz -C ~/Downloads/tmp
sudo -E ~/Downloads/tmp/install
rm -rf ~/Downloads/tmp
```
3. Add paths to `~/.bashrc`
```bash
echo -e "\n# PGI paths" >> ~/.bashrc
echo -e "export PGI=/opt/pgi" >> ~/.bashrc
echo -e "export PATH=/opt/pgi/linux86-64/19.4/bin:$PATH" >> ~/.bashrc
echo -e "export MANPATH=$MANPATH:/opt/pgi/linux86-64/19.4/man" >> ~/.bashrc
echo -e "export LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat" >> ~/.bashrc
```

### Eclipse
1. Download [Eclipse IDE C/C++](https://www.eclipse.org/downloads/download.php?file=/technology/epp/downloads/release/2019-03/R/eclipse-cpp-2019-03-R-linux-gtk-x86_64.tar.gz&mirror_id=1135)
2. Install (assuming file is in *~/Downloads*)
```bash
sudo tar xpfz ~/Downloads/eclipse-cpp-2019-03-R-linux-gtk-x86_64.tar.gz -C /opt
sudo ln -s /opt/eclipse/eclipse /usr/local/bin/eclipse
```
3. Install Nsight Eclipse Plugin -> [more details](https://docs.nvidia.com/cuda/nsightee-plugins-install-guide/index.html)
```bash
bash /usr/local/cuda/bin/nsight_ee_plugins_manage.sh install /opt/eclipse
```
### Docker -> [more details](https://docs.docker.com/install/)
1. Remove older Docker versions
```bash
sudo apt remove docker docker-engine docker.io containerd runc -y
```
2. Install Docker CE -> [more details](https://linuxize.com/post/how-to-install-and-use-docker-on-ubuntu-18-04/)
```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install docker-ce -y
```
3. Enable Docker commands without sudo
```bash
sudo usermod -aG docker $USER
```
4. Log out and back in
- Confirm $USER is in the docker group
```bash
groups
```
> mnicely adm cdrom sudo dip plugdev lpadmin sambashare docker
5. Verify docker runs without sudo
```bash
docker container run hello-world
```
![Docker_Hello_World](https://linuxize.com/post/how-to-install-and-use-docker-on-ubuntu-18-04/docker-hello-world.jpg)

### NVIDIA Docker v2 -> [more details](https://github.com/nvidia/nvidia-docker/wiki/Installation-%28version-2.0%29)
1. Remove nvidia-docker v1, if installed. 
```bash
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge nvidia-docker
```
2. Add nvidia-docker v2 repository
```bash
curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -sL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```
3. Install nvidia-docker v2
```bash
sudo apt-get install nvidia-docker2 -y
```
4. Reload Docker daemon
```bash
sudo pkill -SIGHUP dockerd
```
5. Verify you can launch docker container with access to GPU
```bash
docker run --runtime=nvidia --rm nvcr.io/nvidia/cuda:latest nvidia-smi
```
### PGI Community Edition (Docker Image)

**This create a Docker image containing the PGI CE Compiler.**

1. Download git project
```bash
git clone https://github.com/mnicely/computeWorks_examples.git
cd computeWorks_examples/computeWorks_mm/pgi_build
```
2. Download [PGI CE Compiler](https://www.pgroup.com/support/download_community.php?file=pgi-community-linux-x64)
3. Move PGI tar to Dockerfile directory (assuming file is in *~/Downloads*)
```bash
cp ~/Downloads/pgilinux-2019-194-x86-64.tar.gz .
```
4. Build PGI Docker image
```bash
docker build -t cuda-10.1_ubuntu-18.04_pgi-19.4 -f Dockerfile.cuda-10.1_ubuntu-18.04_pgi-19.4 .
```

### Jupyter Notebook

1. Install PIP package manager
```bash
sudo apt install python-pip
```
2. Install JupyterLab
```bash
sudo -H pip install --upgrade pip
sudo -H pip install jupyter
```

## Usage
### Bare Metal
- This approach requires **PGI Community Edition (Bare Metal)**
1. Download git project
```bash
git clone https://github.com/mnicely/computeWorks_examples.git
cd computeWorks_examples/computeWorks_mm
```
2. Build _computeWorks_mm_ binary
```bash
make
```
3. Run _computeWorks_mm_ <matrixSize | default=1024>
```bash
./computeWorks_mm 128
```

### Docker
- This approach requires **PGI Community Edition (Docker Image)**
1. Download git project
```bash
git clone https://github.com/mnicely/computeWorks_examples.git
cd computeWorks_examples/computeWorks_mm
```
2. Build _computeWorks_mm_ binary
```bash
docker run --runtime=nvidia --rm -v $(pwd):/workspace -w /workspace cuda-10.1_ubuntu-18.04_pgi-19.4:latest make
```
3. Run _computeWorks_mm_ <matrixSize | default=1024>
```bash
docker run --runtime=nvidia --rm -v $(pwd):/workspace -w /workspace cuda-10.1_ubuntu-18.04_pgi-19.4:latest ./computeWorks_mm 128
```

### Eclipse IDE C/C++
Eclipse, with Nsight Eclipse Plugins offers full-featured IDE that provides an all-in-one integrated environment to edit, build, debug and profile CUDA-C applications.

1. Open Eclipse
```bash
eclipse &
```
2. Import Project
  - File -> Import...
  - **Select** -> Git -> Projects from Git -> [_Next >_]
  - **Select Repository Store** -> Clone URI -> [_Next >_]
  - **Source Git Repository** -> URL -> *https://github.com/mnicely/computeWorks_examples* -> [_Next >_]
  - **Branch Selection**  -> [_Next >_]
  - **Local Destination** -> [_Next >_]
  - **Select a wizard to use for importing projects** -> Import existing Eclipse project -> [_Next >_]
  - **Import Projects** -> [_Finish_]
  
## Baremetal
- This approach requires **PGI Community Edition (Bare Metal)**
3. Build Project
  - Right click  _computeWorks_mm_ -> Build Project or Press Ctrl + B
4. Run Project
  - Right click _computeWorks_mm_ -> Run As -> Local C/C++ Application

## Docker container 
- This approach requires **PGI Community Edition (Docker Image)**
3. Point to PGI Docker container
  - Right click  _computeWorks_mm_ -> Properties
  - C/C++ Build -> Settings
  - **Settings** -> Container Settings -> _select_ Build inside Docker Image
  - Image -> cuda-10.1_ubuntu-18.04_pgi-19.4:latest
4. Run Project
  - Right click _computeWorks_mm_ -> Run As -> Local C/C++ Application

### JupyterLab
1. Open `computeWorks_mm.ipynb'
```bash
cd computeWorks_examples/computeWorks_mm/jupyter
jupyter-notebook computeWorks_mm.ipynb
```
