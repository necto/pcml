
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

pred_binary.mat pred_multiclass.mat: models/svmC1.mat models/svmC2.mat models/svmC3.mat models/NeuralNetwork.mat models/RandomForest.mat train/train.mat test/test.mat train/negs3.mat train/negs2.mat train/negs1.mat
	matlab -nodesktop -nosplash -nodisplay -r 'predict;quit'

code/last_updated: Makefile *.m
	mkdir -p code
	cp Makefile code/
	cp *.m code/
	mkdir -p code/models
	touch code/last_updated

report/report.pdf: report/report.tex report/*.tex report/*.bib report/figures/* report/figures/kernel_scale_ber.pdf
	cd report && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex

report/figures/kernel_scale_ber.pdf: SVM2ClassPrediction.m OptimalSVMParams.m train/train.mat
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='optimize_kernel_scale';positiveClass=3;useNegs=false;printFigure=true;useSmalls=false;SVM2ClassPrediction;quit"

small_images: $(small_images_list)

train/negs3.mat train/negs2.mat train/negs1.mat: train models/svmC1.mat models/svmC2.mat models/svmC3.mat
	matlab -nodesktop -nosplash -nodisplay -r 'hardNegativeMining;quit;'
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='produce_model';positiveClass=1;useNegs=true;useSmalls=false;SVM2ClassPrediction;quit"
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='produce_model';positiveClass=2;useNegs=true;useSmalls=false;SVM2ClassPrediction;quit"
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='produce_model';positiveClass=3;useNegs=true;useSmalls=false;SVM2ClassPrediction;quit"

test/test.mat:
	mkdir -p test
	wget "http://cvlabwww.epfl.ch/~cjbecker/tmp/test.mat" -O test/test.mat

train:
	wget "http://cvlabwww.epfl.ch/~cjbecker/pcml_project2/train.tar.gz"
	tar -xvzf train.tar.gz

train/train.mat: train

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

models/NeuralNetwork.mat models/RandomForest.mat: *.m train/train.mat
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;model='trainModels';correctImbalance=false;classificationWithNNandRF;quit"

models/svmC1.mat: *.m
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='produce_model';positiveClass=1;useNegs=false;useSmalls=false;SVM2ClassPrediction;quit"

models/svmC2.mat: *.m
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='produce_model';positiveClass=2;useNegs=false;useSmalls=false;SVM2ClassPrediction;quit"

models/svmC3.mat: *.m
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;task='produce_model';positiveClass=3;useNegs=false;useSmalls=false;SVM2ClassPrediction;quit"
