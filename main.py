import pyomo.environ as pyo
import numpy as np
import pandas as pd


# todo now the designs alternatives set is fixed in dimensions. This means that for example, each part has to have 3 design alternatives.
# todo in the modelling a part can have only 2 alternatives in a 3 dimension set if one is repeated.
# todo it would be best to defined a better set. In the meantime, this works.
# todo maybe define the design alternatives as part of the architecture: arch1 can be the same as arch2 but with a diferent design alternative for a part


# todo remove periods
# add reuse flow
# create instance

# define the size of each set
supplier_number = 3
plants_number = 2
retailers_number = 1
customers_number = 3
collection_centres_number = 3
disassembly_centres_number = 3
remanufacturing_centres_number = 3
parts_number = 2
periods_number = 4
architecture_number = 2
r_imperatives_names = ['refurbishing', 'remanufacturing', 'recycling']
r_imperatives_number = len(r_imperatives_names)
designs_number = 2 # for each part, two designs

big_m = 100000 # todo find a good big m


# create lists of using the number of each set
supplier_list = list(range(0, supplier_number))
plants_list = list(range(0, plants_number))
retailers_list = list(range(0, retailers_number))
customers_list = list(range(0, customers_number))
collection_centres_list = list(range(0,collection_centres_number))
disassembly_centres_list = list(range(0,disassembly_centres_number))
remanufacturing_centres_list = list(range(0, remanufacturing_centres_number))
parts_list = list(range(0, parts_number))
periods_list = list(range(0, periods_number))
architecture_list = list(range(0, architecture_number))
r_imperatives_list = list(range(0, r_imperatives_number))
designs_list = list(range(0, designs_number))


# parameters
np.random.seed(1048596)
# Initialize capacities with random values within a sensible range
suppliers_capacity = np.random.randint(400, 500, (supplier_number, parts_number, periods_number))
plants_capacity = np.random.randint(500, 800, (plants_number, periods_number))
retailers_capacity = np.random.randint(300, 600, (retailers_number, periods_number))
customers_demand = np.random.randint(20, 30, (customers_number, periods_number))
collection_centres_capacity = np.random.randint(100, 500, (collection_centres_number, periods_number))
disassembly_centres_capacity = np.random.randint(100, 400, (disassembly_centres_number, parts_number, periods_number))
remanufacturing_centres_capacity = np.random.randint(100, 300, (remanufacturing_centres_number, parts_number, periods_number))

# Initialize distances with random values representing kilometers, for example
flow_cost_suppliers_plants = np.random.randint(1, 100, (supplier_number, plants_number))
flow_cost_plants_retailers = np.random.randint(1, 100, (plants_number, retailers_number))
flow_cost_retailers_customers = np.random.randint(1, 50, (retailers_number, customers_number))
flow_cost_customers_collection_centres = np.random.randint(1, 50, (customers_number, collection_centres_number))
flow_cost_collection_centres_disassembly = np.random.randint(1, 100, (collection_centres_number, disassembly_centres_number))
flow_cost_collection_centres_plants = np.random.randint(1, 100, (collection_centres_number, plants_number))
flow_cost_disassembly_disposal = np.random.randint(1, 100, (collection_centres_number))
flow_cost_disassembly_remanufacturing_centres = np.random.randint(1, 100, (disassembly_centres_number, remanufacturing_centres_number))
flow_cost_remanufacturing_centre_plants = np.random.randint(1, 100, (remanufacturing_centres_number, plants_number))

df = pd.DataFrame(flow_cost_suppliers_plants)
# df.to_csv('flow_cost_suppliers_plants.csv')
# flow_cost_suppliers_plants = pd.read_csv('flow_cost_suppliers_plants.csv')




# Initialize other parameters with random or specified values
parts_of_architecture = np.random.randint(0, 3, (architecture_number, parts_number)) # rows are the architectures and columns are the part, replaces the r parameter in the model
r_imperatives_of_architecture = np.random.randint(0, 2, (architecture_number, r_imperatives_number)) # rows are architectures and columns the part. It has a value of 1 if the r-imperative is possible with the architecutre
r_imperatives_of_designs = np.random.randint(0, 2, (designs_number, r_imperatives_number)) # rows are designs and columns the part. It has a value of 1 if the r-imperative is possible with the design
designs_of_architecture = np.random.randint(0, 2, (architecture_number, designs_number)) # rows are architecture and columns the design. It has a value of 1 if the design is possible with the architecture
designs_of_parts = np.random.randint(0, 2, (parts_number, designs_number)) # rows are parts and columns the design. It has a value of 1 if the design is possible with the part


alpha = np.random.rand(plants_number, periods_number)
beta = np.random.rand(retailers_number, periods_number)
t_cost = 10  # A given constant for transportation cost or time, for example
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
model.remanufacturing_centres = pyo.Set(initialize=remanufacturing_centres_list) # index r
model.parts = pyo.Set(initialize=parts_list) # index c
model.periods = pyo.Set(initialize=periods_list) # index p
model.architectures = pyo.Set(initialize=architecture_list) # index a, available architectures
model.r_imperatives = pyo.Set(initialize=r_imperatives_list) #  index e, possible r imperatives
model.design_alternatives = pyo.Set(initialize=designs_list) # index s, possible designs that can be used in parts


# Define the  variables of the model

# continuous variables
model.x = pyo.Var(model.suppliers, model.plants,model.parts, model.periods, domain= pyo.NonNegativeReals) # flow from suppliers to plants
model.y = pyo.Var(model.plants,model.retailers,model.periods, domain= pyo.NonNegativeReals) # flow from plants to retailers
model.z = pyo.Var(model.retailers,model.customers,model.periods, domain= pyo.NonNegativeReals) # flow from retailers to customers
model.w = pyo.Var(model.customers,model.collection_centres,model.periods, domain= pyo.NonNegativeReals) # flow from customers to collection centres
model.a = pyo.Var(model.collection_centres, model.plants,model.periods, domain= pyo.NonNegativeReals) # flow from collection centres to plants
model.b = pyo.Var(model.collection_centres, model.disassembly_centres,model.periods, domain=pyo.NonNegativeReals) # flow from collection centres to disassembly centres
model.d = pyo.Var(model.disassembly_centres,model.parts,model.periods, domain=pyo.NonNegativeReals) # flow from from disassembly centres to disposal
model.f = pyo.Var(model.remanufacturing_centres, model.plants, model.parts, model.periods, domain=pyo.NonNegativeReals) # flow from remanufacturing centres to plants


# divided the e flow into e_rf, e_rm and e_r (refurbishing, remanufacturing and recycling)
model.erf = pyo.Var(model.disassembly_centres, model.remanufacturing_centres, model.parts, model.periods, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centres due to refurbishing
model.erm = pyo.Var(model.disassembly_centres, model.remanufacturing_centres, model.parts, model.periods, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centre due to remanufacturing
model.er = pyo.Var(model.disassembly_centres, model.remanufacturing_centres, model.parts, model.periods, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centre due to recycling
# binary variables
model.h = pyo.Var(model.plants,model.periods, domain=pyo.Binary) # if plant j is open at period p
model.g = pyo.Var(model.retailers,model.periods, domain=pyo.Binary) # if retailer r is open at period p



model.ar = pyo.Var(model.architectures, domain= pyo.Binary) # binary, 1 if the product follows architecture a
model.de = pyo.Var(model.design_alternatives, model.parts, domain= pyo.Binary) # 1 if the design alternative s is used in part c

model.rimp = pyo.Var(model.r_imperatives, domain=pyo.Binary) # if r imperative e is possible

# variable to define the save the objective function value (just to have a nice code)
model.objective_variable = pyo.Var(domain=pyo.NonNegativeReals)

# objective function
model.objective = pyo.Objective(expr=model.objective_variable, sense=pyo.minimize)

# Constraints

model.objective_relationship = pyo.ConstraintList()
model.objective_relationship.add(
    # transport costs (the distances matrices must be in cost units)
    sum(model.x[i,j,c,p] * flow_cost_suppliers_plants[i,j] for i in model.suppliers for j in model.plants for c in model.parts for p in model.periods)
    + sum(model.y[j,k,p] * flow_cost_plants_retailers[j,k] for j in model.plants for k in model.retailers for p in model.periods)
    + sum(model.z[k,l,p] * flow_cost_retailers_customers[k,l] for k in model.retailers for l in model.customers for p in model.periods)
    + sum(model.w[l,m,p] * flow_cost_customers_collection_centres[l,m] for l in model.customers for m in model.collection_centres for p in model.periods)
    + sum(model.a[m,j,p] * flow_cost_collection_centres_plants[m,j] for m in model.collection_centres for j in model.plants for p in model.periods)
    + sum(model.b[m,d,p] * flow_cost_collection_centres_disassembly[m,d] for m in model.collection_centres for d in model.disassembly_centres for p in model.periods)
    + sum(model.d[d,c,p] * flow_cost_disassembly_disposal[d] for d in model.disassembly_centres for c in model.parts for p in model.periods)
    + sum(model.erf[d,r,c,p] * flow_cost_disassembly_remanufacturing_centres[d,r] for d in model.disassembly_centres for r in model.remanufacturing_centres for c in model.parts for p in model.periods)
    + sum(model.erm[d,r,c,p] * flow_cost_disassembly_remanufacturing_centres[d, r] for d in model.disassembly_centres for r in model.remanufacturing_centres for c in model.parts for p in model.periods)
    + sum(model.er[d,r,c,p] * flow_cost_disassembly_remanufacturing_centres[d, r] for d in model.disassembly_centres for r in model.remanufacturing_centres for c in model.parts for p in model.periods)

    + sum(model.f[r,j,c,p] * flow_cost_remanufacturing_centre_plants[r,j] for r in model.remanufacturing_centres for j in model.plants for c in model.parts for p in model.periods)

    # opening costs
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
            model.disassembly_centres_capacity_constraints.add(model.d[d,c,p]
                                                               + sum(model.erf[d,r,c,p] for r in model.remanufacturing_centres)
                                                               + sum(model.erm[d,r,c,p] for r in model.remanufacturing_centres)
                                                               + sum(model.er[d,r,c,p] for r in model.remanufacturing_centres)
                                                               <= disassembly_centres_capacity[d,c,p])

# constraint 10
model.remanufacturing_centres_capacity_constraints = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for c in model.parts:
        for p in model.periods:
            model.remanufacturing_centres_capacity_constraints.add(sum(model.f[r,j,c,p] for j in model.plants)
                                                                   <= remanufacturing_centres_capacity[r,c,p])

# constraint 11
model.customer_demand_constraints = pyo.ConstraintList()
for l in model.customers:
    for p in model.periods:
        model.customer_demand_constraints.add(sum(model.z[k,l,p] for k in model.retailers) >= customers_demand[l,p])


# constraint 14
model.plants_flow = pyo.ConstraintList()
for j in model.plants:
    for c in model.parts:
        for p in model.periods:
            if (p>0):
                model.plants_flow.add(sum(model.x[i,j,c,p] for i in model.suppliers)
                                      + sum(model.f[r,j,c,p-1] for r in model.remanufacturing_centres)
                                      + sum(parts_of_architecture[a,c] * model.ar[a] * model.a[m,j,p-1] for a in model.architectures for m in model.collection_centres)
                                      - sum(parts_of_architecture[a,c] * model.ar[a] * model.y[j,k,p] for a in model.architectures for k in model.retailers)
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
            model.disassembly_centres_flow.add(lamda * sum(parts_of_architecture[a,c] * model.ar[a]* model.b[m,d,p] for a in model.architectures for m in model.collection_centres)
                                                - model.d[d,c,p]
                                                == 0 )

# constraint 20
for d in model.disassembly_centres:
    for c in model.parts:
        for p in model.periods:
            model.disassembly_centres_flow.add((1-lamda) * sum(parts_of_architecture[a,c] * model.ar[a] * model.b[m,d,p] for a in model.architectures for m in model.collection_centres)
                                               - sum(model.erf[d,r,c,p] for r in model.remanufacturing_centres)
                                               - sum(model.erm[d,r,c,p] for r in model.remanufacturing_centres)
                                               - sum(model.er[d,r,c,p] for r in model.remanufacturing_centres)
                                               == 0)

# constraint 21
model.remanufacturing_centres_flow = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for c in model.parts:
        for p in model.periods:
            model.remanufacturing_centres_flow.add(sum(model.erf[d,r,c,p] for d in model.disassembly_centres)
                                                   + sum(model.erm[d,r,c,p] for d in model.disassembly_centres)
                                                   + sum(model.er[d,r,c,p] for d in model.disassembly_centres)
                                                   - sum(model.f[r,j,c,p] for j in model.plants)
                                                   == 0)

# constraints 22: we have to select one and only one architecture
model.architecture_limits = pyo.ConstraintList()
model.architecture_limits.add(sum(model.ar[a] for a in model.ar) == 1)

# restrict the refurbishing flow if the r-imperative is not possible. Refurbishing has the index 0
model.refurbishing_possible = pyo.ConstraintList()
for d in model.disassembly_centres:
    for r in model.remanufacturing_centres:
        for c in model.parts:
            for p in model.periods:
                model.refurbishing_possible.add(model.erf[d,r,c,p] <= model.rimp[0])

# restrict the remanufacturing flow if the r-imperative is not possible. Remanufacturing has the index 1
model.remanufacturing_possible = pyo.ConstraintList()
for d in model.disassembly_centres:
    for r in model.remanufacturing_centres:
        for c in model.parts:
            for p in model.periods:
                model.remanufacturing_possible.add(model.erm[d,r,c,p] <= model.rimp[1])

# restrict the recycling flow if the r-imperative is not possible. Recycling has the index 2
model.recycling_possible = pyo.ConstraintList()
for d in model.disassembly_centres:
    for r in model.remanufacturing_centres:
        for c in model.parts:
            for p in model.periods:
                model.recycling_possible.add(model.erf[d,r,c,p] <= model.rimp[2])

# restrict the r-imperative given the selection of architecture
model.r_imperative_possible_architecture = pyo.ConstraintList()
for e in model.r_imperatives:
    for a in model.architectures:
        model.r_imperative_possible_architecture.add(model.rimp[e] * model.ar[a] <= r_imperatives_of_architecture[a,e]) #todo linearize this constraint

# restrict the r-imperative given the selection of design
model.r_imperative_possible_design = pyo.ConstraintList()
for e in model.r_imperatives:
    for s in model.design_alternatives:
        model.r_imperative_possible_design.add(model.rimp[e] * sum(model.de[s,c] for c in model.parts) <= r_imperatives_of_designs[s,e]) #todo linearize this constraint

# relate the part ot a design, only one design can be selected for each part
model.design_to_part = pyo.ConstraintList()
for c in model.parts:
    model.design_to_part.add(sum(model.de[s,c]*designs_of_parts[c,s] for s in model.design_alternatives) == 1)



max_time = 25
solver = 'gurobi'
opt = pyo.SolverFactory(solver)
solution = opt.solve(model)

# form suppliers to plants
for i in model.suppliers:
    for j in model.plants:
        for c in model.parts:
            for p in model.periods:
                if model.x[i,j,c,p].value != 0:
                    print("supplier:",i,"plant:",j,"part:",c,"period:",p)
                    print(model.x[i, j, c, p].value)


# from plants to retailers
for j in model.plants:
    for k in model.retailers:
        for p in model.periods:
                if model.y[j,k,p].value != 0:
                    print("plant:",j,"retailer:",k,"period:",p)
                    print(model.y[j,k,p].value)

# flows from retailers to customers
for k in model.retailers:
    for l in model.customers:
        for p in model.periods:
                if model.z[k,l,p].value != 0:
                    print("retailer:",k,"customer:",l,"period:",p)
                    print(model.z[k,l,p].value)


model.objective_variable.value













