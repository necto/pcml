clear all;
close all;

load('data/regression.mat');

histogram(y_train);
title('Histogram of y\_train.');
hx = xlabel('y\_train');
hy = ylabel('');

set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

disp('printing the figure');
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 20 12]);
set(gcf, 'PaperSize', [20 12]);
print -dpdf 'report/figures/histY.pdf'