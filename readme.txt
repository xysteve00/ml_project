1. put "tiny-imagenet-200" and "miniplaces"  dataset to the folder "ml_project"
2. generate blended images with image_blending_code
   (a): run the following command to generate blended images, and save the blended images to "./img_blend/tiny_blend_image/"
      python find_uap.py resnet50 ../tiny-imagenet-200/train ../tiny-imagenet-200/train  tiny_train.txt tiny_test.txt  ./img_blend/tiny_blend_image/ 'res18' 
   (b) copy "tiny_blend_image" to "tiny-imagenet-200-train" before runing rotation feature decoupling

3. run rotation feature decoupling in "FeatureDecoupling_img_blending" with the following command:
Train a FeatureDecoupling model (with AlexNet architecture) on ImageNet training set:
python main.py --exp=ImageNet_Decoupling_AlexNet --evaluate 0

Train & evaluate linear classifiers for the ImageNet task on the feature maps generated by the convolutional layers (i.e., conv1, conv2, conv3, conv4, and conv5) of the pre-trained FeatureDecoupling model:
python main.py --exp=ImageNet_LinearClassifiers_ImageNet_Decoupling_AlexNet_Features --evaluate 0

Train & evaluate linear classifiers for the Places205 task on the feature maps generated by the convolutional layers (i.e., conv1, conv2, conv3, conv4, and conv5) of the pre-trained FeatureDecoupling model:
python main.py --exp=Places205_LinearClassifiers_ImageNet_Decoupling_AlexNet_Features --evaluate 0

Train & evaluate non-linear classifiers for the ImageNet task on the feature maps generated by the conv4 and conv5 convolutional layers of the pre-trained FeatureDecoupling model:
python main.py --exp=ImageNet_NonLinearClassifiers_ImageNet_Decoupling_AlexNet_Features --evaluate 0

  
