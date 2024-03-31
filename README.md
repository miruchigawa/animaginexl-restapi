# AnimagineXL RestAPI
Simple RestAPI AnimagineXL using diffusers

> *Note:* This project was tested using a Tesla T4 GPU, consuming 7GB-11GB of VRAM. Make sure your GPU has over 8GB of VRAM or use models other than SDXL for low-compute computations.

# Setup

```bash
pip -r REQUIREMENTS.txt
python3 main.py
```

# Reference
[cuDDN cuFFT cuBLAS error](https://github.com/tensorflow/tensorflow/issues/62075)
[Use DPM++ 2M Karras on diffusers issue](https://github.com/huggingface/diffusers/pull/2874)
