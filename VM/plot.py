import os
import matplotlib.pyplot as plt

current_directory = os.getcwd()

data_file = os.path.join(current_directory, "1.txt")
x = []
y = []
with open(data_file, 'r') as file:
    for line in file:
        parts = line.split()
        x.append(int(parts[0]))
        y.append(float(parts[1]))

plt.plot(x, y)
plt.xlabel('День')
plt.ylabel('Выявленные случаи (чел)')
plt.grid(True)

plot_file = os.path.join(current_directory, "graph.png")
plt.savefig(plot_file)
