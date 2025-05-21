# computer-vision-yolo11

The following is example of 'object tracking' in video:

https://github.com/user-attachments/assets/ecb20022-cbfa-4f1d-96ce-eef2f420285b

We used version 11 of the model YOLO (You Only Live Once), open-sourced by Ultralytics (https://huggingface.co/Ultralytics/YOLO11). This model is a convolutional neural network CNN.

The following are examples of 'object detection' in pictures. 

![tmp6yf0_eyz](https://github.com/user-attachments/assets/e935e5b4-347d-4698-96e1-0e531ff4b9a7)

![tmpa3w8y3f7](https://github.com/user-attachments/assets/030157bd-e701-4056-adc5-53429b85fdeb)

![tmp7zd4ruvw](https://github.com/user-attachments/assets/04e512cd-38a3-45b4-a91c-e17a9c46e460)

The following picture, we had used a Diffusion Generative AI model to create the image. Obviously the image has defects. And YOLO is detecting partial birds.

![tmphfolxyyn](https://github.com/user-attachments/assets/1981e1fa-473c-44dd-af2e-56c40198255b)



The following is excerpt of the code we used for object detection:

```
#https://github.com/ultralytics/ultralytics
from ultralytics import YOLO
import time
# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

FlnmsL = !ls -1 /home/lcluser44/Downloads/*jpeg

print( '22' , time.asctime( time.localtime( time.time() ) ))
#
for ii,vv in enumerate(FlnmsL):
    print( '33' , time.asctime( time.localtime( time.time() ) ))
    print(ii,vv)
    Flnm =vv
    # Perform object detection on an image
    results = model( Flnm )  # Predict on an image
    time.sleep(1)
    #
    results[0].show()  # Display results
    print('---------------------\n\n')
    time.sleep(1)
#
print( '998' , time.asctime( time.localtime( time.time() ) ))
```


![tmpv7f13_c5](https://github.com/user-attachments/assets/7aada6d6-109a-4709-ba22-f13ba67e72f3)

![tmpofr6c_0z](https://github.com/user-attachments/assets/3f15fe21-2e19-4eeb-8fc3-854db9c28884)
