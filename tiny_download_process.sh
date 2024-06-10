cd ./clean_dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -d ./tiny tiny-imagenet-200.zip
rm tiny-imagenet-200.zip

cd tiny
mv tiny-imagenet-200/* ./
rm -r tiny-imagenet-200

cd ../
python tiny_data_process.py
echo 'Tiny ImageNet has been downloaded and prossed done'