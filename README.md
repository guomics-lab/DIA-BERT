
<p align="center" style="margin-bottom: 0px !important;">
  <img src=https://github.com/user-attachments/assets/18c5aebc-ae64-4fa4-8952-2dd5c5abc90a width="120" height="120">
</p>

## Description
DIA-BERT: a pre-trained model for data-independent acquisition mass spectrometry-based proteomics data analysis

If you use DIA-BERT in your work, please cite the following publication:

Liu, Z., Liu, P., Sun, Y. et al. DIA-BERT: pre-trained end-to-end transformer models for enhanced DIA proteomics data analysis. Nat Commun 16, 3530 (2025). https://doi.org/10.1038/s41467-025-58866-4.

## Software
The software and manual can be downloaded from https://guomics.com/DIA-BERT/downloads.html.
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


**You need install torch from pytorch (https://pytorch.org/)**. It is advisable to install the entire pytorch package and follow the official installation method provided by pytorch.

Specifically, first select the CUDA version according to your own operating system, and then, based on the CUDA version, choose the corresponding installation command to execute. For example, run "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124".


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


## Frequently asked questions
**Q: Can it analyze semi-specific tryptic samples with DIA?**  
**A:** This model was trained exclusively on fully tryptic peptides (canonical tryptic cleavage rules). We have not yet evaluated its performance on semi-specific tryptic samples (e.g., peptides with missed cleavages or non-canonical termini).
If you’re interested in testing the model on such data, we’d be very keen to learn about your findings! Your feedback would be invaluable for further optimizing the model’s generalizability. Feel free to share any results or observations with us.

**Q: Would it be possible to make the model code available as open source?**  
**A:** As an open-source initiative, our full source code — including the model architecture implementation — is publicly available on this GitHub repository under the license.

**Q: Is it possible to create a spectral library using a human proteome FASTA file to use within DIA-BERT?**  
**A:** DIA-BERT does not support generating a spectral library directly from a FASTA file. However, you can create the spectral library using external tools. The required elements and format for the library are detailed in the user manual, which is available at: https://guomics.com/DIA-BERT/downloads.html.

**Q: What should I do if the file is too large and causes an out-of-memory (OOM) error?**  
**A:** You can try reducing the step_size and batch_size parameters to lower memory usage during training. Alternatively, consider running the process on a GPU with larger memory capacity.

**Q: What information is required in a DIA-MS library for use with DIA-BERT?**  
**A:** The MS DIA spectral library used in DIA-BERT must include the following fields: PeptideSequence, FullUniModPeptideName, PrecursorCharge, PrecursorMz, FragmentMz, iRT, FragmentType, LibraryIntensity, FragmentCharge, ProteinID, and FragmentNumber.
For detailed format specifications, please refer to the user manual, available at: https://guomics.com/DIA-BERT/downloads.html

**Q: What format requirements does the DIA-MS library need to meet?**  
**A:** DIA-BERT supports spectral library files in the following formats: comma-separated (.csv, .txt) and tab-separated (.tsv, .xls, .xlsx).
⚠️ Important: The library must not include non-fragmented precursor ions as fragments. Each fragment ion must originate from the actual fragmentation of the peptide backbone.

**Q: Can I train the model using my own data?**  
**A:** Yes, the model architecture is fully open and publicly available. You can build and train your own model using custom data, and then replace the pre-trained model file in the software with your version.
However, please note that the current version of the software does not support direct training within the application.

**Q: How to install torch required by DIA-BERT?**  
**A:** You can install torch from pytorch (https://pytorch.org/). It is advisable to install the entire pytorch package and follow the official installation method provided by pytorch. Specifically, first select the CUDA version according to your own operating system, and then, based on the CUDA version, choose the corresponding installation command to execute. For example, run "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124".

**Q: Can DIA-BERT process Astral data?**  
**A:** Yes, it can process Astral files. You can select "Other" as the instrument type. However, the identification of Astral files has not been thoroughly evaluated yet, so please use them with caution.

**Q: Is there a way to automatically convert FASTA files or other database formats into the format required by the software?**  
**A:** You can use DIA-NN (https://github.com/vdemichev/DiaNN) to generate a library file in .tsv format.

**Q: If I rent a cloud service with multiple V100 32GB GPU cores (e.g., Nx), do I expect the search speed to be Nx faster? Or, some programming is needed?**  
**A:** GPU Usage Instructions:  
1. If you utilize n GPUs, the processing speed will theoretically be close to n times faster than using a single GPU.  
2. DIA-BERT automatically detects and uses all GPUs with more than 50% available memory, requiring no additional configuration.  
3. If you prefer to manually specify which GPUs to use, please use the following parameter: --gpu_devices [List of GPU indices to be used, separated by commas]  
   Example: --gpu_devices 0,1,2
