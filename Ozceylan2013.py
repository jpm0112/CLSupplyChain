import pyomo.environ as pyo
import numpy as np



# define the size of each set
supplier_number = 3
plants_number = 2
retailers_number = 1
customers_number = 3
collection_centres_number = 3
disassembly_centres_number = 3
refurbishing_centres_number = 3
parts_number = 2
periods_number = 4

# create lists of using the number of each set
supplier_list = list(range(0, supplier_number))
plants_list = list(range(0, plants_number))
retailers_list = list(range(0, retailers_number))
customers_list = list(range(0, customers_number))
collection_centres_list = list(range(0,collection_centres_number))
disassembly_centres_list = list(range(0,disassembly_centres_number))
refurbishing_centres_list = list(range(0,refurbishing_centres_number))
parts_list = list(range(0, parts_number))
periods_list = list(range(0, periods_number))


# parameters

suppliers_capacity = np.zeros((supplier_number,parts_number,periods_number)) # a_icp
plants_capacity = np.zeros((plants_number,periods_number)) #b_jp
retailers_capacity = np.zeros((retailers_number,periods_number)) #c_kp
customers_demand = np.zeros((customers_number,periods_number)) #d_lp
collection_centres_capacity = np.zeros((collection_centres_number,periods_number)) #e_mp
disassembly_centres_capacity = np.zeros((disassembly_centres_number,parts_number, periods_number)) #f_dcp
refurbishing_centres_capacity = np.zeros((refurbishing_centres_number, parts_number,periods_number)) #g_rcp

distance_suppliers_plants = np.zeros((supplier_number,plants_number))
distance_plants_retailers = np.zeros((plants_number,retailers_number))
distance_retailers_customers = np.zeros((retailers_number,customers_number))
distance_customers_collection_centres = np.zeros((customers_number,collection_centres_number))
distance_collection_centres_disassembly = np.zeros((collection_centres_number,disassembly_centres_number))
distance_collection_centres_plants = np.zeros((collection_centres_number,plants_number))
distance_disassembly_disposal = np.zeros((collection_centres_number))
distance_disassembly_refurbishing_centres = np.zeros((disassembly_centres_number,refurbishing_centres_number))
distance_refurbishing_centre_plants = np.zeros((refurbishing_centres_number,plants_number))

r = np.zeros((parts_number))
alpha = np.zeros((plants_number,periods_number))
beta = np.zeros((retailers_number,periods_number))
t_cost = 10
s = np.zeros((supplier_number, parts_number))
w = np.zeros((refurbishing_centres_number, parts_number))
h_limit = np.zeros((periods_number))
g_limit = np.zeros((periods_number))
nu = 0.2
sigma = 0.1
lamda = 0.1


# Initialize capacities with random values within a sensible range
suppliers_capacity = np.random.randint(400, 500, (supplier_number, parts_number, periods_number))
plants_capacity = np.random.randint(500, 800, (plants_number, periods_number))
retailers_capacity = np.random.randint(300, 600, (retailers_number, periods_number))
customers_demand = np.random.randint(20, 30, (customers_number, periods_number))
collection_centres_capacity = np.random.randint(100, 500, (collection_centres_number, periods_number))
disassembly_centres_capacity = np.random.randint(100, 400, (disassembly_centres_number, parts_number, periods_number))
refurbishing_centres_capacity = np.random.randint(100, 300, (refurbishing_centres_number, parts_number, periods_number))

# Initialize distances with random values representing kilometers, for example
distance_suppliers_plants = np.random.randint(1, 100, (supplier_number, plants_number))
distance_plants_retailers = np.random.randint(1, 100, (plants_number, retailers_number))
distance_retailers_customers = np.random.randint(1, 50, (retailers_number, customers_number))
distance_customers_collection_centres = np.random.randint(1, 50, (customers_number, collection_centres_number))
distance_collection_centres_disassembly = np.random.randint(1, 100, (collection_centres_number, disassembly_centres_number))
distance_collection_centres_plants = np.random.randint(1, 100, (collection_centres_number, plants_number))
distance_disassembly_disposal = np.random.randint(1, 100, (collection_centres_number))
distance_disassembly_refurbishing_centres = np.random.randint(1, 100, (disassembly_centres_number, refurbishing_centres_number))
distance_refurbishing_centre_plants = np.random.randint(1, 100, (refurbishing_centres_number, plants_number))

# Initialize other parameters with random or specified values
r_parts = np.random.randint(1, 10, (parts_number))
alpha = np.random.rand(plants_number, periods_number)
beta = np.random.rand(retailers_number, periods_number)
t_cost = 10  # A given constant for transportation cost or time, for example
s = np.random.randint(1, 20, (supplier_number, parts_number)) # Cost of parts from suppliers
w = np.random.randint(1, 20, (refurbishing_centres_number, parts_number)) # Cost of refurbished parts
h_limit = np.random.randint(1, 5, (periods_number)) # Total supply chain capacity limit for a period
g_limit = np.random.randint(1, 5, (periods_number)) # Total demand limit for a period
nu = 0.2  # Recycling rate, for example
sigma = 0.1  # Breakage rate, for example
lamda = 0.1  # Loss rate, for example








# start the model
model = pyo.ConcreteModel()

# create the sets of the model using the  pyomo
model.suppliers = pyo.Set(initialize=supplier_list) # index i
model.plants = pyo.Set(initialize=plants_list) # index j
model.retailers = pyo.Set(initialize=retailers_list) # index k
model.customers = pyo.Set(initialize=customers_list) # index l
model.collection_centres = pyo.Set(initialize=collection_centres_list) # index m
model.disassembly_centres = pyo.Set(initialize=disassembly_centres_list) # index d
model.refurbishing_centres = pyo.Set(initialize=refurbishing_centres_list) # index r
model.parts = pyo.Set(initialize=parts_list) # index c
model.periods = pyo.Set(initialize=periods_list) # index p


# Define the  variables of the model

# continuous variables
model.x = pyo.Var(model.suppliers, model.plants,model.parts, model.periods, domain= pyo.NonNegativeReals)
model.y = pyo.Var(model.plants,model.retailers,model.periods, domain= pyo.NonNegativeReals)
model.z = pyo.Var(model.retailers,model.customers,model.periods, domain= pyo.NonNegativeReals)
model.w = pyo.Var(model.customers,model.collection_centres,model.periods, domain= pyo.NonNegativeReals)
model.a = pyo.Var(model.collection_centres, model.plants,model.periods, domain= pyo.NonNegativeReals)
model.b = pyo.Var(model.collection_centres, model.disassembly_centres,model.periods, domain=pyo.NonNegativeReals)
model.d = pyo.Var(model.disassembly_centres,model.parts,model.periods, domain=pyo.NonNegativeReals)
model.e = pyo.Var(model.disassembly_centres,model.refurbishing_centres,model.parts, model.periods, domain=pyo.NonNegativeReals)
model.f = pyo.Var(model.refurbishing_centres,model.plants,model.parts, model.periods, domain=pyo.NonNegativeReals)

# binary variables
model.h = pyo.Var(model.plants,model.periods, domain=pyo.Binary)
model.g = pyo.Var(model.retailers,model.periods, domain=pyo.Binary)

# variable to define the objective function (just for a nice code)
model.objective_variable = pyo.Var(domain=pyo.NonNegativeReals)

# objective function
model.objective = pyo.Objective(expr=model.objective_variable, sense=pyo.minimize)

# Constraints

model.objective_relationship = pyo.ConstraintList()
model.objective_relationship.add(
    sum(model.x[i,j,c,p] * distance_suppliers_plants[i,j] for i in model.suppliers for j in model.plants for c in model.parts for p in model.periods)
    + sum(model.y[j,k,p] * distance_plants_retailers[j,k]  for j in model.plants for k in model.retailers for p in model.periods)
    + sum(model.z[k,l,p] * distance_retailers_customers[k,l] for k in model.retailers for l in model.customers for p in model.periods)
    + sum(model.w[l,m,p] * distance_customers_collection_centres[l,m] for l in model.customers for m in model.collection_centres for p in model.periods)
    + sum(model.a[m,j,p] * distance_collection_centres_plants[m,j] for m in model.collection_centres for j in model.plants for p in model.periods)
    + sum(model.b[m,d,p] * distance_collection_centres_disassembly[m,d] for m in model.collection_centres for d in model.disassembly_centres for p in model.periods)
    + sum(model.d[d,c,p] * distance_disassembly_disposal[d] for d in model.disassembly_centres for c in model.parts for p in model.periods)
    + sum(model.e[d,r,c,p] * distance_disassembly_refurbishing_centres[d,r] for d in model.disassembly_centres for r in model.refurbishing_centres for c in model.parts for p in model.periods)
    + sum(model.f[r,j,c,p] * distance_refurbishing_centre_plants[r,j] for r in model.refurbishing_centres for j in model.plants for c in model.parts for p in model.periods)

    + sum(model.x[i,j,c,p] * s[i,c] for i in model.suppliers for j in model.plants for c in model.parts for p in model.periods)
    + sum(model.f[r,j,c,p] * w[r,c] for r in model.refurbishing_centres for c in model.parts for j in model.plants for p in model.periods)
    + sum(model.h[j,p] * alpha[j,p] for j in model.plants for p in model.periods)
    + sum(model.g[k,p] * beta[k,p] for k in model.retailers for p in model.periods)

    <= model.objective_variable)


# constraint 5
model.capacity_suppliers_constraints = pyo.ConstraintList()
for i in model.suppliers:
    for c in model.parts:
        for p in model.periods:
            model.capacity_suppliers_constraints.add(sum(model.x[i,j,c,p] for j in model.plants) <= suppliers_capacity[i,c,p])

# constraint 6
model.capacity_plants_constraints = pyo.ConstraintList()
for j in model.plants:
    for p in model.periods:
        model.capacity_plants_constraints.add(sum(model.y[j,k,p] for k in model.retailers) <= plants_capacity[j,p]*model.h[j,p])

# constraint 7
model.capacity_retailers_constraints = pyo.ConstraintList()
for k in model.retailers:
    for p in model.periods:
        model.capacity_retailers_constraints.add(sum(model.z[k,l,p] for l in model.customers) <= retailers_capacity[k,p] * model.g[k,p])

# constraint 8
model.capacity_collection_centres_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for p in model.periods:
        model.capacity_collection_centres_constraints.add(sum(model.a[m,j,p] for j in model.plants)
                                              + sum(model.b[m,d,p] for d in model.disassembly_centres )
                                              <= collection_centres_capacity[m,p])

# constraint 9
model.disassembly_centres_capacity_constraints = pyo.ConstraintList()
for d in model.disassembly_centres:
    for c in model.parts:
        for p in model.periods:
            model.disassembly_centres_capacity_constraints.add(model.d[d,c,p] + sum(model.e[d,r,c,p] for r in model.refurbishing_centres) <= disassembly_centres_capacity[d,c,p])

# constraint 10
model.refurbishing_centres_capacity_constraints = pyo.ConstraintList()
for r in model.refurbishing_centres:
    for c in model.parts:
        for p in model.periods:
            model.refurbishing_centres_capacity_constraints.add(sum(model.f[r,j,c,p] for j in model.plants)
                                                    <= refurbishing_centres_capacity[r,c,p])

# constraint 11
model.customer_demand_constraints = pyo.ConstraintList()
for l in model.customers:
    for p in model.periods:
        model.customer_demand_constraints.add(sum(model.z[k,l,p] for k in model.retailers) >= customers_demand[l,p])

# constraint 12
model.plant_opening_limit = pyo.ConstraintList()
for p in model.periods:
    model.plant_opening_limit.add(sum(model.h[j,p] for j in model.plants) <= h_limit[p])

# constraint 13
model.retailer_opening_limit = pyo.ConstraintList()
for p in model.periods:
    model.retailer_opening_limit.add(sum(model.g[k,p] for k in model.retailers) <= g_limit[p])

# constraint 14
model.plants_flow = pyo.ConstraintList()
for j in model.plants:
    for c in model.parts:
        for p in model.periods:
            if (p>0):
                model.plants_flow.add(sum(model.x[i,j,c,p] for i in model.suppliers)
                                      + sum(model.f[r,j,c,p-1] for r in model.refurbishing_centres)
                                      + r_parts[c] * sum(model.a[m,j,p-1] for m in model.collection_centres)
                                      - r_parts[c] * sum(model.y[j,k,p] for k in model.retailers)
                                      == 0 )
# constraint 15
model.retailers_flow = pyo.ConstraintList()
for k in model.retailers:
    for p in model.periods:
        model.retailers_flow.add(sum(model.y[j,k,p] for j in model.plants)
                                 - sum(model.z[k,l,p] for l in model.customers)
                                 == 0)

# constraint 16

model.customers_flow = pyo.ConstraintList()
for l in model.customers:
    for p in model.periods:
        model.customers_flow.add(nu * sum(model.z[k,l,p] for k in model.retailers)
                                 - sum(model.w[l,m,p] for m in model.collection_centres)
                                 == 0)

# constraint 17
model.collections_centres_flow = pyo.ConstraintList()
for m in model.collection_centres:
    for p in model.periods:
        model.collections_centres_flow.add(sigma * sum(model.w[l,m,p] for l in model.customers)
                                 - sum(model.a[m,j,p] for j in model.plants)
                                 == 0)

# constraint 18
for m in model.collection_centres:
    for p in model.periods:
        model.collections_centres_flow.add((1-sigma) * sum(model.w[l,m,p] for l in model.customers)
                                - sum(model.b[m,d,p] for d in model.disassembly_centres)
                                == 0 )

# constraint 19
model.disassembly_centres_flow = pyo.ConstraintList()
for d in model.disassembly_centres:
    for c in model.parts:
        for p in model.periods:
            model.disassembly_centres_flow.add(r_parts[c] * lamda * sum(model.b[m,d,p] for m in model.collection_centres)
                                                - model.d[d,c,p]
                                                == 0 )

# constraint 20
for d in model.disassembly_centres:
    for c in model.parts:
        for p in model.periods:
            model.disassembly_centres_flow.add(r_parts[c] * (1-lamda) * sum(model.b[m,d,p] for m in model.collection_centres)
                                               - sum(model.e[d,r,c,p] for r in model.refurbishing_centres)
                                               == 0)

# constraint 21
model.refurbishing_centres_flow = pyo.ConstraintList()
for r in model.refurbishing_centres:
    for c in model.parts:
        for p in model.periods:
            model.refurbishing_centres_flow.add(sum(model.e[d,r,c,p] for d in model.disassembly_centres)
                                                - sum(model.f[r,j,c,p] for j in model.plants)
                                                == 0 )

# constraints 22 and 23 are not necessary if the nature of the variables are defined when they are created



max_time = 25
solver = 'gurobi'
opt = pyo.SolverFactory(solver)
opt.options['TimeLimit'] = max_time
solution = opt.solve(model)


for i in model.suppliers:
    for j in model.plants:
        for c in model.parts:
            for p in model.periods:
                print(model.x[i,j,c,p].value)

model.objective_variable.value













