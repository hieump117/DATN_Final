import os.path

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import GetImageUploadSerializers
import cv2
from .BlurDetection import main, FocusMask
import cloudinary.uploader
from . import utils
from .NAFNet.basicsr.models import create_model
from .NAFNet.basicsr.utils.options import parse
class PredictImageUploadAPIView(APIView):
    def post(self,request):
        serializer = GetImageUploadSerializers(data=request.data)
        if serializer.is_valid():
            # Lấy ảnh từ dữ liệu đầu vào và xử lý ảnh ở đây
            # Sau đó trả về kết quả xử lý

            # Ví dụ: Lưu ảnh vào thư mục media và trả về URL
            image = serializer.validated_data['image']
            image_name = image.name
            with open('myapp/images/' + image_name, 'wb') as img_file:
                for chunk in image.chunks():
                    img_file.write(chunk)

            img_path = 'myapp/images/'+image_name
            img = cv2.imread(img_path)
            img_fft, val, blurry = main.blur_detector(img)
            msk, val, blur = FocusMask.blur_mask(img)
            _, image_encoded = cv2.imencode('.jpg', msk)
            result = cloudinary.uploader.upload(image_encoded.tobytes())
            return Response({'blur':blurry,'image':result['url']}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DeblurImageAPIView(APIView):
    def post(self,request):
        serializer = GetImageUploadSerializers(data=request.data)
        if serializer.is_valid():
            # Lấy ảnh từ dữ liệu đầu vào và xử lý ảnh ở đây
            # Sau đó trả về kết quả xử lý

            # Ví dụ: Lưu ảnh vào thư mục media và trả về URL
            image = serializer.validated_data['image']
            image_name = image.name
            input_path = 'myapp/images/deblur/input/' + image_name
            output_path = 'myapp/images/deblur/output/' + image_name
            if not os.path.exists(input_path):
                with open('myapp/images/deblur/input/' + image_name, 'wb') as img_file:
                    for chunk in image.chunks():
                        img_file.write(chunk)
                opt_path = 'myapp/NAFNet/options/test/REDS/NAFNet-width64.yml'
                opt = parse(opt_path, is_train=False)
                opt['dist'] = False
                NAFNet = create_model(opt)
                img_input = utils.imread(input_path)
                inp = utils.img2tensor(img_input)
                utils.single_image_inference(NAFNet, inp, output_path)
            # img_output = utils.imread(output_path)
            result = cloudinary.uploader.upload(output_path)
            return Response({'image': result['url']}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
