import csv


class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'a')
        self.fname = fname

    def log(self, fields):
        csv.writer(self.f).writerow(fields)
        self.f.flush()
        self.f.close()
        self.f = open(self.fname, 'a')


class FigLogger(object):
    def __init__(self, fig, base_ax, title):
        self.colors = ['tab:red', 'tab:blue']
        self.labels = ['Loss (entropy)', 'Error']
        self.markers = ['d', '.']
        self.axes = [base_ax, base_ax.twinx()]
        base_ax.set_xlabel('Epochs')
        base_ax.set_title(title)
        
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(self.labels[i], color=self.colors[i])
            ax.tick_params(axis='y', labelcolor=self.colors[i])

        self.reset()
        self.fig = fig
        
    def log(self, args):
        for i, arg in enumerate(args[-2:]):
            self.curves[i].append(arg)
            x = list(range(len(self.curves[i])))
            self.axes[i].plot(x, self.curves[i], self.colors[i], marker=self.markers[i])
        self.axes[1].set_ylim(0, 1.05)
            
        self.fig.canvas.draw()
        
    def reset(self):
        for ax in self.axes:
            for line in ax.lines:
                line.remove()
        self.curves = [[], []]