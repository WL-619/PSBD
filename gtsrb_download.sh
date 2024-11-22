# https://github.com/SCLBD/BackdoorBench/blob/main/sh/gtsrb_download.sh
# please download the following files and put them in ./clean_dataset folder
# ! Do not use the torch.datasets.GTSRB
wget -P ./clean_dataset https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip --no-check-certificate
wget -P ./clean_dataset https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip --no-check-certificate
wget -P ./clean_dataset https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip --no-check-certificate
mkdir ./clean_dataset/gtsrb;
mkdir ./clean_dataset/gtsrb/Train;
mkdir ./clean_dataset/gtsrb/Test;
mkdir ./clean_dataset/temps;
unzip ./clean_dataset/GTSRB_Final_Training_Images.zip -d ./clean_dataset/temps/Train;
unzip ./clean_dataset/GTSRB_Final_Test_Images.zip -d ./clean_dataset/temps/Test;
mv ./clean_dataset/temps/Train/GTSRB/Final_Training/Images/* ./clean_dataset/gtsrb/Train;
mv ./clean_dataset/temps/Test/GTSRB/Final_Test/Images/* ./clean_dataset/gtsrb/Test;
unzip ./clean_dataset/GTSRB_Final_Test_GT.zip -d ./clean_dataset/gtsrb/Test/;
rm -r ./clean_dataset/temps;
rm ./clean_dataset/*.zip;
echo "Download Completed";
