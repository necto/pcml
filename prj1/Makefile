
test:
	matlab -nodesktop -nosplash -nojvm -r 'testCode;quit'

report/figures/penLLmisses.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;stage='penLogReg';classification;quit"

report/figures/polyLogLoss.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;stage='polynomial';classification;quit"

report/figures/histY.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;regression;quit"

report/figures/CorrelationXY.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;regression;quit"

report/figures/X58vsY.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;regression;quit"

report/figures/X43vsY.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;regression;quit"

report/figures/ridgeRegLoss.pdf: *.m
	mkdir -p report/figures
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;stage='ridgeReg';regression;quit"

predictions_classification.csv test_errors_classification.csv: *.m
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;stage='logReg';classification;quit"

predictions_regression.csv test_errors_regression.csv: *.m
	matlab -nodesktop -nosplash -nodisplay -r "reportMode=true;stage='leastSq';regression;quit"

report/report.pdf: report/report.tex report/figures/penLLmisses.pdf report/figures/polyLogLoss.pdf report/figures/histY.pdf report/figures/CorrelationXY.pdf report/figures/X58vsY.pdf report/figures/X43vsY.pdf report/figures/ridgeRegLoss.pdf
	cd report && pdflatex report.tex

istanbul.zip: predictions_regression.csv test_errors_regression.csv predictions_classification.csv test_errors_classification.csv report/report.pdf *.m
	mkdir -p submission
	cp leastSquaresGD.m submission/
	cp computeGradient.m submission/
	cp leastSquares.m submission/
	cp ridgeRegression.m submission/
	cp logisticRegression.m submission/
	cp logisticRegLoss.m submission/
	cp sigmoid.m submission/
	cp log1PlusExp.m submission/
	cp logisticNewton.m submission/
	cp logisticRegLossNewton.m submission/
	cp penLogisticRegression.m submission/
	cp penLogisticRegLoss.m submission/
	cp predictions_regression.csv submission/
	cp test_errors_regression.csv submission/
	cp predictions_classification.csv submission/
	cp test_errors_classification.csv submission/
	cp report/report.pdf submission/
	zip -r istanbul.zip submission

package: istanbul.zip

clean:
	rm predictions_classification.csv
	rm report/figures/*.pdf
	rm report/*.log report/*.out report/*.pdf
