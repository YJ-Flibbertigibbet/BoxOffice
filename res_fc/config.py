from torchvision import transforms

#=======================================CONFIG======================================

#图片文件位置
picFolder="box_office/data/pic"
photosFolder="box_office/data/photos"

#评论文件
commentFile="box_office/data/filtered_comments.xlsx"

#数据文件
boxDataFile="box_office/data/train_ceshi.xlsx"
textBoxDataFile="box_office/data/test_ceshi.xlsx"
#对应图片索引列名
picDataId="id"

#训练
batchSize=64
epochs=100
lr=0.001
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])