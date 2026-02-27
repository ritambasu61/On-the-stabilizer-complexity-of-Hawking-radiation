import numpy as np
import numpy as np
import matplotlib.pyplot as plt

db = 47
dr = 47


# Load data from .npy files (each is a 2D array: [[time, y_value], ...])
data_numerical = np.load('/home/ritam.basu/Desktop/hh_coordinate/Db=' + str(db) + 'Dr=' + str(dr) + '.npy')
#data_numerical = data_numerical[::2]  # shape: (N, 2)
data_analytical = np.load('/home/ritam.basu/Desktop/hh_coordinate/final_Db=' + str(db) + 'Dr=' + str(dr) + '.npy')  # shape: (N, 2)
data_analytical = data_analytical[::2]  # shape: (N, 2)
print('ana==========',data_analytical)
print('num==========',data_numerical)
# Extract time and y-values
time_numerical = data_numerical[:, 0]
y_numerical = data_numerical[:, 1]
y_analytical = data_analytical


time = time_numerical  # or time_analytical (they're the same now)

# Avoid division by zero
epsilon = 1e-10
safe_y_numerical = np.where(np.abs(y_numerical) < epsilon, epsilon, y_numerical)

# Compute relative error
relative_error = (y_numerical - y_analytical) / safe_y_numerical

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(time, relative_error, label='Relative Error', color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Relative Error')
plt.title('Relative Error vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/ritam.basu/Desktop/hh_coordinate/final/101/error/error_final_Db=' + str(db) + 'Dr=' + str(dr) + '.pdf')
plt.show()


