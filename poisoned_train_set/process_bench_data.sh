cd ./cifar10
unzip -d ./issba_0_1 cifar10_issba_0_1.zip
unzip -d ./issba_0_05 cifar10_issba_0_05.zip
unzip -d ./trojannn_0_1 cifar10_trojannn_0_1.zip
unzip -d ./trojannn_0_05 cifar10_trojannn_0_05.zip
rm *.zip

cd ../tiny
unzip -d ./lc_backdoor_train tiny_lc_backdoor_train.zip
unzip -d ./lc_backdoor_test tiny_lc_backdoor_test.zip
unzip -d ./issba_0_1 tiny_issba_0_1.zip
unzip -d ./issba_0_05 tiny_issba_0_05.zip
unzip -d ./trojannn_0_1 tiny_trojannn_0_1.zip
unzip -d ./trojannn_0_05 tiny_trojannn_0_05.zip
rm *.zip

echo "Bench data has been processed done"