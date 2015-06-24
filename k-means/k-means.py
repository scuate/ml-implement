##plotting of clustering by k-means
import numpy as np
import matplotlib.pyplot as plt
from algorithm import assignCtr,PI,NI,kmeans,kmeans_plot,cost_plot

##the data file is provided by Prof. Marina Meila
##set the number of clusters K, parameter c for PI, iterations T
data = np.loadtxt("cluster5-data1000.dat")
k = 4
c = 2
T = 100

### plot the trajectories of the K centers, compare power initialization with naive initialization
x = data[:,0]
y = data[:,1]
plt.plot(x,y, ".", label="data points")
cs = PI(data,k,c)
kmeans_plot(data,k,cs,T,"red","PI")
cs = NI(data,k)
kmeans_plot(data,k,cs,T,"yellow","NI")
plt.legend(loc = "lower left")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

### mark the data points by their cluster assignments and mark the final position of the centers
cs = PI(data,k,c)
# cs = NI(data,k)
final_ctrs = kmeans(data,k,cs,T)
assignment = assignCtr(data, final_ctrs)
i=0
colors = ["blue","green","cyan","pink"]
for v in assignment.itervalues():
    pts = np.array(v)
    x = pts[:,0]
    y = pts[:,1]
    plt.plot(x,y,".",color=colors[i])
    i+=1
 
x = final_ctrs[:,0]
y = final_ctrs[:,1]
plt.plot(x,y,"ro",color="red",label="centers by PI")
 
plt.legend(loc = "lower left")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

###compare the log-cost over iterations between power initialization and naive initialization
cs = PI(data,k,c)
cost_plot(data,k,cs,T,"red","PI")
cs = NI(data,k)
cost_plot(data,k,cs,T,"blue","NI")
plt.legend(loc = "upper right")
plt.xlabel('iterations')
plt.ylabel('log-cost')
plt.show()