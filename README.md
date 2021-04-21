# DeepLeague
![Alt text](demo.gif)
## Requirements:
  ./yolov5/requirements.txt
  scikit-image==0.17.2
  vietocr==0.3.5
  imgaug==0.4.0
  gradio
## Pretrained models:
- Download pretrained model của CRAFT tại:
  https://drive.google.com/file/d/1ZqFMCTi1qdGgPSwVblXr4nQQRRNaXtnY/view?usp=sharing
  https://drive.google.com/file/d/1LBu06Sn-Peom9aqyIz8YzYtw6Xaas0G6/view?usp=sharing
  đặt vào thư mục ./CRAFTpytorch/basenet/
 - Download pretrained model Yolov5 tại:
  https://drive.google.com/file/d/1Cqw3iPGFkgPKZW7o7r33JYGICo4oy_Ot/view?usp=sharing
  đặt vào thư mục ./yolov5/runs/train/
## Inference:
+ Inference với commandline:
  - Chạy: python inference.py
  - Nhập vào video_path cần xử lý.
 + Inference với UI: (yêu cầu Gradio)
  - Chạy: python inferenceUI.py
## Note:
- Hiện chương trình chỉ có thể xử lý video đầu vào với độ phân giải 1920x1080!
# References:
https://github.com/farzaa/DeepLeague
