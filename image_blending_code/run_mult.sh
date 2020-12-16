#python find_uap.py resnet50 ./warplane/ ./warplane/  tar_list.txt ntar_list.txt  ./img_blend/tnt/
#python find_uap.py resnet50 ./data/n04552348/ ./data/n04552348/  target.txt target.txt  ./img_blend/
#python find_uap.py resnet50 './data/Scrap2/' './data/Scrap2/'  scrap_test_target.txt scrap_test_target.txt  ./img_blend/
#python find_uap.py resnet50 '' ''  './data/3.txt' './data/3.txt'  ./img_blend/
python find_uap_mult.py resnet50 ./data/Scrap2/ ./data/Scrap2/  scrap_train.txt scrap_test.txt  ./img_blend/smv0/ 'res18'
#python find_uap.py resnet50 ./data/open_image_val/ ./data/open_image_val/  open_image_list.txt open_image_list.txt  ./img_blend/feature/

