
<p align="center" style="margin-bottom: 0px !important;">
  <img src=https://github.com/user-attachments/assets/3ef95b67-5244-49df-840c-ba9e3e9a3b84 width="120" height="120">
</p>

## Description
DIA-BERT: a pre-trained model for data-independent acquisition mass spectrometry-based proteomics data analysis

## Software
The software and manual can be downloaded from the website https://guomics.com/DIA-BERT/.
On Windows systems, download and unzip the zip file. Click on DIA-BERT.exe to run without installation. 
On Linux, download the file from the release. DIA-BERT runs install-free and requires no additional configuration of the environment. 

## Installation
If you want to use DIA-BERT by source code, you can install python and install requirements package.

### Prerequisites
Please make sure you have a valid installation of conda or miniconda. We recommend setting up miniconda as described on their website.

```shell
git clone https://github.com/guomics-lab/DIA-BERT.git
cd DIA-BERT
```

```shell
conda create -n DIA-BERT python=3.10
conda activate DIA-BERT
```

```shell
pip install -r requirements.txt
```

Run GUI
```shell
python main_applet.py
```

Windows command-line run
```shell
python main_win.py

```
Linux command-line run
```shell
python main_linux.py
```

## Hardware Requirements:
•	Operating System: Supports both Windows and Linux operating systems.

•	Processor: A dual-core processor is recommended, but it can run on a single-core processor.

•	Memory: 40GB or more is recommended. If the mass spectrometry files or library files to be identified are large, it is advised to use more memory.

•	Storage: At least 100GB of available hard disk space is recommended.

•	Graphics Card: A 40GB NVIDIA GPU with CUDA support or a V100 32GB GPU is recommended.

## License
This software is licensed under a custom license that allows academic use but prohibits commercial use. For more details, see the LICENSE file.

## Contact
For any questions or licensing inquiries, please contact:
Dr Guo
E-mail: guotiannan@westlake.edu.cn
www.guomics.com
