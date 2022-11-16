function plotConf(C)
img = (C./sum(C,1));
z = 15;
img = imresize(img, z, 'nearest');
imshow(img)
img(isnan(img)) = 0;
fig = imshow(1-img);
colormap hot
c = colorbar;
c.YDir = 'reverse';
axis on
fig.YData = [0 1];
fig.XData = [0 1];
set(gca, 'xticklabels', {'','0','1'});
set(gca, 'yticklabels', {'','0','1'});
xlim([0 1]);
ylim([0 1]);
c.TickLabels = 100:-50:0;
xlabel('True Labels')
ylabel('Predicted Labels')
ylabel(c, 'Percentage of Classification')
title('Confusion Matrix');
grid on
% grid minor