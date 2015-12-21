
images_list=$(wildcard train/imgs/*.jpg)
small_images_list=$(subst imgs,small_imgs,$(images_list)) 

submission: istanbul.zip

istanbul.zip: report/report.pdf pred_binary.mat pred_multiclass.mat code/last_updated
	mkdir -p submission
	cp report/report.pdf submission/report.pdf
	cp -r code submission/
	cp pred_binary.mat submission/pred_binary.mat
	cp pred_multiclass.mat submission/pred_multiclass.mat
	zip -r -j istanbul.zip submission
	rm -r submission

code/last_updated: Makefile *.m
	cp Makefile code/
	cp *.m code/
	touch code/last_updated

report/report.pdf: report/report.tex report/*.tex report/*.bib report/figures/*
	cd report && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex

small_images: $(small_images_list)

train/negs3.mat train/negs2.mat train/negs1.mat: train models/svmC1.mat models/svmC2.mat models/svmC3.mat
	matlab -nodesktop -nosplash -nodisplay -r 'hardNegativeMining;quit;'

test/test.mat:
	mkdir -p test
	wget "http://cvlabwww.epfl.ch/~cjbecker/tmp/test.mat" -O test/test.mat

train:
	wget "http://cvlabwww.epfl.ch/~cjbecker/pcml_project2/train.tar.gz"
	tar -xvzf train.tar.gz

train/small_imgs: train
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

