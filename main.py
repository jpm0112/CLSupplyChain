import pyomo.environ as pyo
import numpy as np
import pandas as pd


# todo now the designs alternatives set is fixed in dimensions. This means that for example, each part has to have 3 design alternatives.
# todo in the modelling a part can have only 2 alternatives in a 3 dimension set if one is repeated.
# todo it would be best to defined a better set. In the meantime, this works.
# todo maybe define the design alternatives as part of the architecture: arch1 can be the same as arch2 but with a diferent design alternative for a part

# collection_processing_cost = pd.read_csv('collection_processing_cost.csv', delimiter=';', header=None)
#
# flow_cost_collection_centres_plants = pd.read_csv('flow_cost_collection_centres_plants.csv', delimiter=';', header = None)
# flow_cost_collection_reprocessing = pd.read_csv('flow_cost_collection_reprocessing.csv', delimiter=';', header = None)
# flow_cost_reprocessing_disposal = pd.read_csv('flow_cost_reprocessing_disposal.csv', delimiter=';', header = None)
# flow_cost_reprocessing_plants = pd.read_csv('flow_cost_reprocessing_plants.csv', delimiter=';', header = None)
flow_cost_suppliers_plants = pd.read_csv('flow_cost_suppliers_plants.csv', delimiter=';', header = None)
#
# product_design = pd.read_csv('product_design.csv', delimiter=';', header = None)
# production_cost = pd.read_csv('production_cost.csv', delimiter=';', header = None)
# R_imperatives_cost = pd.read_csv('R_imperatives_cost.csv', delimiter=';', header = None)
# R_imperatives_possibility = pd.read_csv('R_imperatives_possibility.csv', delimiter=';', header = None)
virgin_material_purchasing_cost = pd.read_csv('virgin_material_purchasing_cost.csv', delimiter=';', header = None)



# todo remove periods
# add reuse flow
# create instance
# add index to X for the design alternative
# add architecture index to capacities
# different cost for archtecture in dissambly

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
suppliers_capacity = np.random.randint(400, 500, (supplier_number, parts_number))
plants_capacity = np.random.randint(500, 800, (plants_number))
# retailers_capacity = np.random.randint(300, 600, (retailers_number))
retailer_demand = np.random.randint(2, 3, (retailers_number))
collection_centres_capacity = np.random.randint(100, 500, (collection_centres_number))
disassembly_centres_capacity = np.random.randint(100, 400, (disassembly_centres_number, parts_number))
remanufacturing_centres_capacity = np.random.randint(100, 300, (remanufacturing_centres_number, parts_number))

flow_cost_suppliers_plants = np.random.randint(1, 100, (supplier_number, plants_number)) # suppliers as rows, plants as columns
flow_cost_plants_retailers = np.random.randint(1, 100, (plants_number, retailers_number))
flow_cost_retailers_customers = np.random.randint(1, 50, (retailers_number, customers_number))
flow_cost_customers_collection_centres = np.random.randint(1, 50, (customers_number, collection_centres_number))
flow_cost_collection_centres_disassembly = np.random.randint(1, 100, (collection_centres_number, disassembly_centres_number))
flow_cost_collection_centres_plants = np.random.randint(1, 100, (collection_centres_number, plants_number))
flow_cost_disassembly_disposal = np.random.randint(1, 100, (collection_centres_number))
flow_cost_disassembly_remanufacturing_centres = np.random.randint(1, 100, (disassembly_centres_number, remanufacturing_centres_number))
flow_cost_remanufacturing_centre_plants = np.random.randint(1, 100, (remanufacturing_centres_number, plants_number))






# Initialize other parameters with random or specified values
parts_of_architecture = np.random.randint(0, 3, (architecture_number, parts_number)) # rows are the architectures and columns are the part, replaces the r parameter in the model
r_imperatives_of_architecture = np.random.randint(0, 2, (architecture_number, r_imperatives_number)) # rows are architectures and columns the part. It has a value of 1 if the r-imperative is possible with the architecutre
r_imperatives_of_designs = np.random.randint(0, 2, (designs_number, r_imperatives_number)) # rows are designs and columns the part. It has a value of 1 if the r-imperative is possible with the design
designs_of_architecture = np.random.randint(0, 2, (architecture_number, designs_number)) # rows are architecture and columns the design. It has a value of 1 if the design is possible with the architecture
designs_of_parts = np.random.randint(1, 2, (parts_number, designs_number)) # rows are parts and columns the design. It has a value of 1 if the design is possible with the part


alpha = np.random.rand(plants_number)
beta = np.random.rand(retailers_number)
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
# model.customers = pyo.Set(initialize=customers_list) # index l
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
model.x = pyo.Var(model.suppliers, model.plants, model.parts, domain= pyo.NonNegativeReals) # flow from suppliers to plants
model.y = pyo.Var(model.plants,model.retailers, domain= pyo.NonNegativeReals) # flow from plants to retailers
# model.z = pyo.Var(model.retailers,model.customers, domain= pyo.NonNegativeReals) # flow from retailers to customers
model.w = pyo.Var(model.retailers,model.collection_centres, domain= pyo.NonNegativeReals) # flow from customers to collection centres
model.a = pyo.Var(model.collection_centres, model.plants, domain= pyo.NonNegativeReals) # flow from collection/dissasembly centres to plants
model.b = pyo.Var(model.collection_centres, model.retailers, domain= pyo.NonNegativeReals) # flow from collection/dissasembly centres to retailers
model.d = pyo.Var(model.disassembly_centres, domain=pyo.NonNegativeReals) # flow from from collection/dissasembly centres to disposal
model.f = pyo.Var(model.collection_centres, model.remanufacturing_centres, model.parts, domain=pyo.NonNegativeReals) # flow from collection/dissasembly centres to remanufacturing centres

# divided the e flow into e_rf, e_rm and e_r (refurbishing, remanufacturing and recycling)
model.erf = pyo.Var(model.disassembly_centres, model.remanufacturing_centres, model.parts, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centres due to refurbishing
model.erm = pyo.Var(model.disassembly_centres, model.remanufacturing_centres, model.parts, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centre due to remanufacturing
model.er = pyo.Var(model.disassembly_centres, model.remanufacturing_centres, model.parts, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centre due to recycling
# binary variables # todo add these kind of variables for the rest of the relevant nodes
model.h = pyo.Var(model.plants, domain=pyo.Binary) # if plant j is open at period p
model.g = pyo.Var(model.retailers, domain=pyo.Binary) # if retailer r is open at period p



model.ar = pyo.Var(model.architectures, domain= pyo.Binary) # binary, 1 if the product follows architecture a
model.de = pyo.Var(model.design_alternatives, model.parts, domain= pyo.Binary) # 1 if the design alternative s is used for part c

model.rimp = pyo.Var(model.r_imperatives, domain=pyo.Binary) # if r imperative e is possible

# variable to define the save the objective function value (just to have a nice code)
model.objective_variable = pyo.Var(domain=pyo.NonNegativeReals)

# objective function
model.objective = pyo.Objective(expr=model.objective_variable, sense=pyo.minimize)

# Constraints

model.objective_relationship = pyo.ConstraintList()
model.objective_relationship.add(
    # transport costs (the distances matrices must be in cost units)
    sum(model.x[i,j,c] * flow_cost_suppliers_plants[i,j] for i in model.suppliers for j in model.plants for c in model.parts)
    + sum(model.y[j,k] * flow_cost_plants_retailers[j,k] for j in model.plants for k in model.retailers )
    + sum(model.z[k,l] * flow_cost_retailers_customers[k,l] for k in model.retailers for l in model.customers )
    + sum(model.w[l,m] * flow_cost_customers_collection_centres[l,m] for l in model.customers for m in model.collection_centres )
    + sum(model.a[m,j] * flow_cost_collection_centres_plants[m,j] for m in model.collection_centres for j in model.plants )
    + sum(model.b[m,d] * flow_cost_collection_centres_disassembly[m,d] for m in model.collection_centres for d in model.disassembly_centres )
    + sum(model.d[d,c] * flow_cost_disassembly_disposal[d] for d in model.disassembly_centres for c in model.parts )
    + sum(model.erf[d,r,c] * flow_cost_disassembly_remanufacturing_centres[d,r] for d in model.disassembly_centres for r in model.remanufacturing_centres for c in model.parts )
    + sum(model.erm[d,r,c] * flow_cost_disassembly_remanufacturing_centres[d, r] for d in model.disassembly_centres for r in model.remanufacturing_centres for c in model.parts )
    + sum(model.er[d,r,c] * flow_cost_disassembly_remanufacturing_centres[d, r] for d in model.disassembly_centres for r in model.remanufacturing_centres for c in model.parts )
    + sum(model.f[r,j,c] * flow_cost_remanufacturing_centre_plants[r,j] for r in model.remanufacturing_centres for j in model.plants for c in model.parts )

    # opening costs
    + sum(model.h[j] * alpha[j] for j in model.plants )
    + sum(model.g[k] * beta[k] for k in model.retailers )

    <= model.objective_variable)


# # constraint 1: capacity of suppliers
model.capacity_suppliers_constraints = pyo.ConstraintList()
for i in model.suppliers:
    for c in model.parts:
            model.capacity_suppliers_constraints.add(sum(model.x[i,j,c] for j in model.plants) <= suppliers_capacity[i,c])


# constraint 2: capacity of plants
model.capacity_plants_constraints = pyo.ConstraintList()
for j in model.plants:
        model.capacity_plants_constraints.add(sum(model.y[j,k] for k in model.retailers) <= plants_capacity[j]*model.h[j])

# constraint 3: demand of retailers
model.demand_retailers_constraints = pyo.ConstraintList()
for k in model.retailers:
        model.demand_retailers_constraints.add(sum(model.w[k,m] for m in model.collection_centres) >= retailer_demand[k])

# constraint 4: capacity of the collection/disassembly centre
model.capacity_collection_centres_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for s in model.design_alternatives:
        for c in model.parts:
            for a in model.architectures:
                model.capacity_collection_centres_constraints.add((sum(model.a[m, j] for j in model.plants)
                                                                   + sum(model.b[m, k] for k in model.retailers)
                                                                   + model.d[m]) * model.ar[a] * parts_of_architecture[a,c]
                                                                  + sum(model.f[m, r, c] for r in
                                                                        model.remanufacturing_centres)
                                                                  <= collection_centres_capacity[m, c])


# constraint 5: capacity of the refurbishing centres
model.remanufacturing_centres_capacity_constraints = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for c in model.parts:
            model.remanufacturing_centres_capacity_constraints.add(sum(model.erf[r,j,c] for j in model.plants)
                                                               + sum(model.erm[r,j,c] for j in model.plants)
                                                               + sum(model.er[r,j,c] for j in model.plants)
                                                               <= remanufacturing_centres_capacity[r,c])



# constraint 11
model.customer_demand_constraints = pyo.ConstraintList()
for l in model.customers:
        model.customer_demand_constraints.add(sum(model.z[k,l] for k in model.retailers) >= retailer_demand[l])


# constraint 14
model.plants_flow = pyo.ConstraintList()
for j in model.plants:
    for c in model.parts:
        for s in model.design_alternatives:
                model.plants_flow.add(sum(model.x[i,j,s] for i in model.suppliers)
                                      + sum(model.f[r,j,c] for r in model.remanufacturing_centres)
                                      + sum(parts_of_architecture[a,c] * model.ar[a] * model.a[m,j] for a in model.architectures for m in model.collection_centres)
                                      - sum(parts_of_architecture[a,c] * model.ar[a] * model.y[j,k] for a in model.architectures for k in model.retailers)
                                      == 0 )
# constraint 15
model.retailers_flow = pyo.ConstraintList()
for k in model.retailers:
    for p in model.periods:
        model.retailers_flow.add(sum(model.y[j,k] for j in model.plants)
                                 - sum(model.z[k,l] for l in model.customers)
                                 == 0)

# constraint 16

model.customers_flow = pyo.ConstraintList()
for l in model.customers:
        model.customers_flow.add(nu * sum(model.z[k,l] for k in model.retailers)
                                 - sum(model.w[l,m] for m in model.collection_centres)
                                 == 0)

# constraint 17
model.collections_centres_flow = pyo.ConstraintList()
for m in model.collection_centres:
        model.collections_centres_flow.add(sigma * sum(model.w[l,m] for l in model.customers)
                                 - sum(model.a[m,j] for j in model.plants)
                                 == 0)

# constraint 18
for m in model.collection_centres:
        model.collections_centres_flow.add((1-sigma) * sum(model.w[l,m] for l in model.customers)
                                - sum(model.b[m,d] for d in model.disassembly_centres)
                                == 0 )

# constraint 19
model.disassembly_centres_flow = pyo.ConstraintList()
for d in model.disassembly_centres:
    for c in model.parts:
            model.disassembly_centres_flow.add(lamda * sum(parts_of_architecture[a,c] * model.ar[a]* model.b[m,d] for a in model.architectures for m in model.collection_centres)
                                                - model.d[d,c]
                                                == 0 )

# constraint 20
for d in model.disassembly_centres:
    for c in model.parts:
            model.disassembly_centres_flow.add((1-lamda) * sum(parts_of_architecture[a,c] * model.ar[a] * model.b[m,d] for a in model.architectures for m in model.collection_centres)
                                               - sum(model.erf[d,r,c] for r in model.remanufacturing_centres)
                                               - sum(model.erm[d,r,c] for r in model.remanufacturing_centres)
                                               - sum(model.er[d,r,c] for r in model.remanufacturing_centres)
                                               == 0)

# constraint 21
model.remanufacturing_centres_flow = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for c in model.parts:
            model.remanufacturing_centres_flow.add(sum(model.erf[d,r,c] for d in model.disassembly_centres)
                                                   + sum(model.erm[d,r,c] for d in model.disassembly_centres)
                                                   + sum(model.er[d,r,c] for d in model.disassembly_centres)
                                                   - sum(model.f[r,j,c] for j in model.plants)
                                                   == 0)

# constraints 22: we have to select one and only one architecture
model.architecture_limits = pyo.ConstraintList()
model.architecture_limits.add(sum(model.ar[a] for a in model.ar) == 1)

# restrict the refurbishing flow if the r-imperative is not possible. Refurbishing has the index 0
model.refurbishing_possible = pyo.ConstraintList()
for d in model.disassembly_centres:
    for r in model.remanufacturing_centres:
        for c in model.parts:
                model.refurbishing_possible.add(model.erf[d,r,c] <= model.rimp[0])

# restrict the remanufacturing flow if the r-imperative is not possible. Remanufacturing has the index 1
model.remanufacturing_possible = pyo.ConstraintList()
for d in model.disassembly_centres:
    for r in model.remanufacturing_centres:
        for c in model.parts:
                model.remanufacturing_possible.add(model.erm[d,r,c] <= model.rimp[1])

# restrict the recycling flow if the r-imperative is not possible. Recycling has the index 2
model.recycling_possible = pyo.ConstraintList()
for d in model.disassembly_centres:
    for r in model.remanufacturing_centres:
        for c in model.parts:
                model.recycling_possible.add(model.erf[d,r,c] <= model.rimp[2])

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
                # if model.x[i,j,c].value != 0:
                    print("supplier:",i,"plant:",j,"part:",c)
                    print(model.x[i, j, c].value)


# from plants to retailers
for j in model.plants:
    for k in model.retailers:

                # if model.y[j,k].value != 0:
                    print("plant:",j,"retailer:",k)
                    print(model.y[j,k].value)

# flows from retailers to customers
for k in model.retailers:
    for l in model.customers:

                # if model.z[k,l].value != 0:
                    print("retailer:",k,"customer:",l)
                    print(model.z[k,l].value)


model.objective_variable.value





#
# # Define function to convert numpy arrays to pandas DataFrame, save as CSV, and reload as DataFrame
# def numpy_to_csv_and_reload(data, filename):
#     # Convert to DataFrame
#     df = pd.DataFrame(data)
#     # Save to CSV
#     csv_path = filename+'.csv'
#     df.to_csv(csv_path, index=False)
#     # Reload from CSV
#     reloaded_df = pd.read_csv(csv_path)
#     return reloaded_df, csv_path
#
# # Convert and reload each array as per the given example
# flow_cost_suppliers_plants_df, flow_cost_suppliers_plants_path = numpy_to_csv_and_reload(flow_cost_suppliers_plants, 'flow_cost_suppliers_plants')
# flow_cost_plants_retailers_df, flow_cost_plants_retailers_path = numpy_to_csv_and_reload(flow_cost_plants_retailers, 'flow_cost_plants_retailers')
# flow_cost_retailers_customers_df, flow_cost_retailers_customers_path = numpy_to_csv_and_reload(flow_cost_retailers_customers, 'flow_cost_retailers_customers')
# flow_cost_customers_collection_centres_df, flow_cost_customers_collection_centres_path = numpy_to_csv_and_reload(flow_cost_customers_collection_centres, 'flow_cost_customers_collection_centres')
# flow_cost_collection_centres_disassembly_df, flow_cost_collection_centres_disassembly_path = numpy_to_csv_and_reload(flow_cost_collection_centres_disassembly, 'flow_cost_collection_centres_disassembly')
# flow_cost_collection_centres_plants_df, flow_cost_collection_centres_plants_path = numpy_to_csv_and_reload(flow_cost_collection_centres_plants, 'flow_cost_collection_centres_plants')
# flow_cost_disassembly_disposal_df, flow_cost_disassembly_disposal_path = numpy_to_csv_and_reload(flow_cost_disassembly_disposal, 'flow_cost_disassembly_disposal')
# flow_cost_disassembly_remanufacturing_centres_df, flow_cost_disassembly_remanufacturing_centres_path = numpy_to_csv_and_reload(flow_cost_disassembly_remanufacturing_centres, 'flow_cost_disassembly_remanufacturing_centres')
# flow_cost_remanufacturing_centre_plants_df, flow_cost_remanufacturing_centre_plants_path = numpy_to_csv_and_reload(flow_cost_remanufacturing_centre_plants, 'flow_cost_remanufacturing_centre_plants')
#
# # List of all saved file paths
# saved_files_paths = [
#     flow_cost_suppliers_plants_path,
#     flow_cost_plants_retailers_path,
#     flow_cost_retailers_customers_path,
#     flow_cost_customers_collection_centres_path,
#     flow_cost_collection_centres_disassembly_path,
#     flow_cost_collection_centres_plants_path,
#     flow_cost_disassembly_disposal_path,
#     flow_cost_disassembly_remanufacturing_centres_path,
#     flow_cost_remanufacturing_centre_plants_path,
# ]
#
# saved_files_paths







