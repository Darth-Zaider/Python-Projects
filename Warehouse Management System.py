# # Warehouse Optimization system

# This warehouse optimization system is based on a linear programming model that solves the warehouse location problem with transportation costs, fixed costs, warehouse capacities, and customer demands. 
# Utilizing the pulp library for Python it minimizes the total cost of transportation and fixed warehouse costs while satisfying certain constraints: A warehouse cannot deliver more goods than its capacity and each customer's demand must be satisfied.

# # Model overview
# Decision Variables:
# xi,j: Units transported from supplier i to warehouse j.
# yj: Binary variable indicating whether a warehouse is operational at location j (1 if yes, 0 if no).
 

# Objective Function:
# Total Cost = ΣTi,j * xi,j + ΣFj * yj
# Where:
# 	Ci,j = Transportation cost per unit of goods
# 	Fj = Fixed Cost for operating warehouse j
 

# Constraints:
# Demand: Ensure customer demand is met.
# Capacity: Ensure warehouse capacity isn’t exceeded.
# Binary Constraints: Warehouses are either operational or not.

# In[2]:


import pulp
import numpy as np

# Dataset
transportation_costs = np.array([[20, 24, 11, 25], [28, 27, 82, 83], [74, 97, 71, 96]])  # Cost matrix - represents the transportation cost from warehouses to customers. There are 3 warehouses (rows) and 4 customers (columns). The values are the cost to transport goods from a warehouse to a customer.
fixed_costs = [100, 200, 150, 180]  # Fixed costs for each warehouse
warehouse_capacity = [500, 400, 600, 700]  # Warehouse capacities
customer_demand = [300, 400, 500, 600, 200]  # Customer demands


# optimization model
model = pulp.LpProblem("Warehouse_Location_Optimization", pulp.LpMinimize)

# decision variables
x = pulp.LpVariable.dicts("Transport", ((i, j) for i in range(3) for j in range(4)), lowBound=0, cat='Continuous')
y = pulp.LpVariable.dicts("Warehouse", range(4), cat='Binary')

# objective function
model += pulp.lpSum(transportation_costs[i][j] * x[i, j] for i in range(3) for j in range(4)) + \
         pulp.lpSum(fixed_costs[j] * y[j] for j in range(4))

# constraints
for j in range(4):
    model += pulp.lpSum(x[i, j] for i in range(3)) <= warehouse_capacity[j] * y[j]  # Capacity constraint

for i in range(3):
    model += pulp.lpSum(x[i, j] for j in range(4)) >= customer_demand[i]  # Demand satisfaction constraint


# solve the model
model.solve()

#  print results
print("Status:", pulp.LpStatus[model.status])
print("Optimal Cost:", pulp.value(model.objective))

for var in model.variables():
    print(f"{var.name} = {var.varValue}")


# From the output we can see that the result is an optimal configuration with a total cost of 50650.0 currency units. 
# the 'Transport' output indicates customer(0) receives 300.0 units of goods from warehouse(2), customer(1) receives 400.0 units of goods from warehouse(1), and customer(2) receives 200.0 units of goods from warehouse(0) and 300.0 units of goods from warehouse(2). 
# This result satifies both the demand and capacity constraints. 
# The 'Warehouse' output indicates which warehouses need to  be operational for the model to work, and shows that Warehouse(3) is not necessary / could be closed.
