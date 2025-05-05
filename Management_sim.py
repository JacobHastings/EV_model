from Vehicle_class import Vehicle
from Charger_class import Charger
from Manager_class import Manager
import matplotlib.pyplot as plt
import numpy as np

V1 = Vehicle()
V1.index = 1
V1.battery_SOC = 80
V1.update_capacity()
V1.update_log()

V2 = Vehicle()
V2.index = 2
V2.battery_SOC = 40
V2.update_capacity()
V2.update_log()

V3 = Vehicle()
V3.index = 3
V3.update_log()
V3.maximum_charge_rate = 800

V4 = Vehicle()
V4.index = 4
V4.battery_SOC = 100
V4.update_capacity()
V4.update_log()

V5 = Vehicle()
V5.index = 5
V5.update_log()

# Extra Vehicle added later
V6 = Vehicle()
V6.index = 6
V6.battery_SOC = 10
V6.update_capacity()
V6.update_log()


C1 = Charger()
C1.maximum_load = 1000

C2 = Charger()
C2.maximum_load = 1000

M = Manager()

M.chargers.append(C1)
M.chargers.append(C2)

M.vehicles.append(V1)
M.vehicles.append(V2)
M.vehicles.append(V3)
M.vehicles.append(V4)
M.vehicles.append(V5)

M.to_charge.extend(M.vehicles)

sim_time = 0
interval = 900
done = False

while not done:
    M.simulate(interval)
    sim_time += interval
    # Determine if everything is done charging
    done = True
    if not M.to_charge:
        for C in M.chargers:
            if C.occupied:
                done = False
    else:
        done = False
    # try adding V6 to queue 32 hours in
    if sim_time == (3600*32):
        M.to_charge.append(V6)
    
M.simulate(interval)
plot_time = []
plot_load = []
for X in M.load_log:
    plot_time.append(X[0])
    plot_load.append(X[1])
    
plot_time = np.array(plot_time)
plot_load = np.array(plot_load)

plt.plot(plot_time/3600,plot_load)
plt.xlabel("Time (Hours)")
plt.ylabel("Load (Watts)")
plt.grid()
plt.show()