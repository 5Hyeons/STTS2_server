Original repo: [StyleTTS2](https://github.com/yl4579/StyleTTS2)

Paper: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)

## Environment setup
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/5Hyeons/STTS2_server.git
cd STTS2_server
```
3. If you don’t have PyTorch installed, follow the instructions [here](https://pytorch.org/get-started/locally/) to install the appropriate versions of torch and torchaudio.

4. Install python requirements: 
```bash
pip install -r requirements.txt
```
  - Note: You must have `cuDNN` installed to use `onnxruntime-gpu`.
5. Download the models [here](https://drive.google.com/drive/folders/1-Ec0OnZ-KHLPiFfRALdjbrXvf0TRU1jW?usp=sharing) and place them in the STTS2_server/Models directory.
  
## Run server
- To run the server using a .pth model:
```bash
python server.py
```
- To Run the server using an ONNX model:
```bash
python server_onnx.py
```