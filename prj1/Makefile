
test:
	matlab -nodesktop -nosplash -nojvm -r 'testCode;quit'

report/figures/penLLmisses.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r 'reportMode=true;classification;quit'

report/report.pdf: report/report.tex
	cd report
	pdflatex report.tex
	cd ..

istanbul.zip: predictions_regression.csv test_errors_regression.csv predictions_classification.csv test_errors_classification.csv report.pdf *.m
	mkdir -p submission
	cp leastSquaresGD.m submission/
	cp leastSquares.m submission/
	cp ridgeRegression.m submission/
	cp logisticRegression.m submission/
	cp logisticRegLoss.m submission/
	cp logisticNewton.m submission/
	cp logisticRegLossNewton.m submission/
	cp penLogisticRegression.m submission/
	cp predictions_regression.csv submission/
	cp test_errors_regression.csv submission/
	cp predictions_classification.csv submission/
	cp test_errors_classification.csv submission/
	cp report.pdf submission/
	zip -r istanbul.zip submission

package: istanbul.zip

clean:
	rm predictions_classification.csv
