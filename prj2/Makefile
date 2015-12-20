
images_list=$(wildcard train/imgs/*.jpg)
small_images_list=$(subst imgs,small_imgs,$(images_list)) 

small_images: $(small_images_list)

train/small_imgs:
	mkdir -p train/small_imgs

train/small_imgs/%.jpg : train/imgs/%.jpg train/small_imgs
	convert $< -resize 25% $@

train/small.mat: small_train_features.m $(small_images_list)
	matlab -nodesktop -nosplash -nodisplay -r 'small_train_features;quit;'


overfeat.tgz:
	wget "http://cilvr.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-v04-2.tgz" -O overfeat.tgz

# You also need to install openblas (e.g. sudo apt-get install libopenblas-base)
overfeat: overfeat.tgz
	tar -xvzf overfeat.tgz
	cd overfeat && python download_weights.py

