import pyomo.environ as pyo
import numpy as np
import pandas as pd



# define the size of each set

# watch out when defining a value to one. The saving fucntion of the csv file may save a list rather than matrix.
# comment the loading section if you want to run a test instance defining one of the number values to one.
supplier_number = 20
plants_number = 20
retailers_number = 10
collection_centres_number = 10
disassembly_centres_number = 10
remanufacturing_centres_number = 2
components_number = 3
periods_number = 4
architecture_number = 2
r_imperatives_names = ['refurbishing', 'remanufacturing', 'recycling', 'reusing', 'repacking'] # assuming A is repacking and B is reusing. If not, switch names here.
r_imperatives_number = len(r_imperatives_names)
designs_number = 2 # for each part, two designs

big_m = 100000 # todo find a good big m

alpha = 0.2
beta = 0.3
nu = 0.2  # percentage collected from the retailers
sigma = 0.1  # percentage from the collected product that is good enough for reusing or repackaging
lamda = 0.1  # percentage of the collected product that has to be disposed


supplier_list = list(range(supplier_number))
plants_list = list(range(plants_number))
retailers_list = list(range(retailers_number))
collection_centres_list = list(range(collection_centres_number))
disassembly_centres_list = list(range(disassembly_centres_number))
remanufacturing_centres_list = list(range(remanufacturing_centres_number))
components_list = list(range(components_number))
periods_list = list(range(periods_number))
architecture_list = list(range(architecture_number))
r_imperatives_list = list(range(r_imperatives_number))
designs_list = list(range(designs_number))




# TO GENERATE CSVs, COMMENT THIS SECTION IF YOU WANT TO USE AN ISNTANCE FROM FILES
# parameters
np.random.seed(1048596)
# Initialize capacities with random values within a sensible range
suppliers_capacity = np.random.randint(400, 500, (supplier_number, components_number))
np.savetxt('suppliers_capacity.csv', suppliers_capacity, delimiter=",")
plants_capacity = np.random.randint(500, 800, (plants_number))
np.savetxt('plants_capacity.csv', plants_capacity, delimiter=",")
retailer_demand = np.random.randint(200, 300, (retailers_number, periods_number))
np.savetxt('retailer_demand.csv', retailer_demand, delimiter=",")
collection_centres_capacity = np.random.randint(300, 500, (collection_centres_number, components_number))
np.savetxt('collection_centres_capacity.csv', collection_centres_capacity, delimiter=",")
disassembly_centres_capacity = np.random.randint(300, 400, (disassembly_centres_number, components_number))
np.savetxt('disassembly_centres_capacity.csv', disassembly_centres_capacity, delimiter=",")
remanufacturing_centres_capacity = np.random.randint(300, 400, (remanufacturing_centres_number, components_number))
np.savetxt('remanufacturing_centres_capacity.csv', remanufacturing_centres_capacity, delimiter=",")

flow_cost_suppliers_plants = np.random.randint(1, 100,(supplier_number, plants_number))  # suppliers as rows, plants as columns
np.savetxt('flow_cost_suppliers_plants.csv', flow_cost_suppliers_plants, delimiter=",")
purchase_cost_suppliers = np.random.randint(1, 100, (components_number, designs_number))  # suppliers as rows, plants as columns
np.savetxt('purchase_cost_suppliers.csv', purchase_cost_suppliers, delimiter=",")
flow_cost_plants_retailers = np.random.randint(1, 100, (plants_number, retailers_number))
np.savetxt('flow_cost_plants_retailers.csv', flow_cost_plants_retailers, delimiter=",")
flow_cost_retailers_collection_centres = np.random.randint(1, 50, (retailers_number, collection_centres_number))
np.savetxt('flow_cost_retailers_collection_centres.csv', flow_cost_retailers_collection_centres, delimiter=",")
flow_cost_collection_retailer = np.random.randint(1, 50, (collection_centres_number, retailers_number))
np.savetxt('flow_cost_collection_retailer.csv', flow_cost_collection_retailer, delimiter=",")

flow_cost_collection_centres_remanufacturing = np.random.randint(1, 100, (collection_centres_number, remanufacturing_centres_number))
np.savetxt('flow_cost_collection_centres_remanufacturing.csv', flow_cost_collection_centres_remanufacturing,delimiter=",")

flow_cost_collection_centres_plants = np.random.randint(1, 100, (collection_centres_number, plants_number))
np.savetxt('flow_cost_collection_centres_plants.csv', flow_cost_collection_centres_plants, delimiter=",")
flow_cost_disassembly_disposal = np.random.randint(1, 100, (collection_centres_number))
np.savetxt('flow_cost_disassembly_disposal.csv', flow_cost_disassembly_disposal, delimiter=",")

flow_cost_remanufacturing_refurbishing = np.random.randint(1, 100, (remanufacturing_centres_number, plants_number))
np.savetxt('flow_cost_remanufacturing_refurbishing.csv', flow_cost_remanufacturing_refurbishing, delimiter=",")
flow_cost_remanufacturing_reclycling = np.random.randint(1, 100, (remanufacturing_centres_number, plants_number))
np.savetxt('flow_cost_remanufacturing_reclycling.csv', flow_cost_remanufacturing_reclycling, delimiter=",")
flow_cost_remanufacturing_remanufacturing = np.random.randint(1, 100, (remanufacturing_centres_number, plants_number))
np.savetxt('flow_cost_remanufacturing_remanufacturing.csv', flow_cost_remanufacturing_remanufacturing, delimiter=",")

opening_cost_collection = np.random.randint(1, 100, (collection_centres_number))
np.savetxt('opening_cost_collection.csv', opening_cost_collection, delimiter=",")
opening_cost_reprocessing = np.random.randint(1, 100, (remanufacturing_centres_number))
np.savetxt('opening_cost_reprocessing.csv', opening_cost_reprocessing, delimiter=",")
opening_cost_supplier = np.random.randint(1, 100, (supplier_number))
np.savetxt('opening_cost_supplier.csv', opening_cost_supplier, delimiter=",")

# Initialize other parameters with random or specified values
bill_of_materials = np.random.randint(1, 2, (architecture_number, components_number))  # rows are the architectures and columns are the part, replaces the r parameter in the model
np.savetxt('bill_of_materials.csv', bill_of_materials, delimiter=",")
r_imperatives_of_architecture = np.random.randint(0, 2, (architecture_number,r_imperatives_number))  # rows are architectures and columns the part. It has a value of 1 if the r-imperative is possible with the architecutre
np.savetxt('r_imperatives_of_architecture.csv', r_imperatives_of_architecture, delimiter=",")
r_imperatives_of_designs = np.random.randint(0, 2, (designs_number, r_imperatives_number))  # rows are designs and columns the part. It has a value of 1 if the r-imperative is possible with the design
np.savetxt('r_imperatives_of_designs.csv', r_imperatives_of_designs, delimiter=",")
designs_of_architecture = np.random.randint(0, 2, (architecture_number,designs_number))  # rows are architecture and columns the design. It has a value of 1 if the design is possible with the architecture
np.savetxt('designs_of_architecture.csv', designs_of_architecture, delimiter=",")
designs_of_components = np.random.randint(1, 2, (components_number, designs_number))  # rows are parts and columns the design. It has a value of 1 if the design is possible with the part
np.savetxt('designs_of_components.csv', designs_of_components, delimiter=",")




# TO LOAD CSVs
suppliers_capacity = np.loadtxt('suppliers_capacity.csv', delimiter=',')
plants_capacity = np.loadtxt('plants_capacity.csv', delimiter=',')
retailer_demand = np.loadtxt('retailer_demand.csv', delimiter=',')
collection_centres_capacity = np.loadtxt('collection_centres_capacity.csv', delimiter=',')
disassembly_centres_capacity = np.loadtxt('disassembly_centres_capacity.csv', delimiter=',')
remanufacturing_centres_capacity = np.loadtxt('remanufacturing_centres_capacity.csv', delimiter=',')

flow_cost_suppliers_plants = np.loadtxt('flow_cost_suppliers_plants.csv', delimiter=',')
purchase_cost_suppliers = np.loadtxt('purchase_cost_suppliers.csv', delimiter=',')
flow_cost_plants_retailers = np.loadtxt('flow_cost_plants_retailers.csv', delimiter=',')
flow_cost_retailers_collection_centres = np.loadtxt('flow_cost_retailers_collection_centres.csv', delimiter=',')
flow_cost_collection_retailer = np.loadtxt('flow_cost_collection_retailer.csv', delimiter=',')

flow_cost_collection_centres_remanufacturing = np.loadtxt('flow_cost_collection_centres_remanufacturing.csv',delimiter=',')
flow_cost_collection_centres_plants = np.loadtxt('flow_cost_collection_centres_plants.csv', delimiter=',')
flow_cost_disassembly_disposal = np.loadtxt('flow_cost_disassembly_disposal.csv', delimiter=',')
flow_cost_remanufacturing_refurbishing = np.loadtxt('flow_cost_remanufacturing_refurbishing.csv', delimiter=',')
flow_cost_remanufacturing_reclycling = np.loadtxt('flow_cost_remanufacturing_reclycling.csv', delimiter=',')
flow_cost_remanufacturing_remanufacturing = np.loadtxt('flow_cost_remanufacturing_remanufacturing.csv', delimiter=',')

opening_cost_collection = np.loadtxt('opening_cost_collection.csv', delimiter=',')
opening_cost_reprocessing = np.loadtxt('opening_cost_reprocessing.csv', delimiter=',')
opening_cost_supplier = np.loadtxt('opening_cost_supplier.csv', delimiter=',')

bill_of_materials = np.loadtxt('bill_of_materials.csv', delimiter=',')

r_imperatives_of_architecture = np.loadtxt('r_imperatives_of_architecture.csv', delimiter=',')
r_imperatives_of_designs = np.loadtxt('r_imperatives_of_designs.csv', delimiter=',')

designs_of_architecture = np.loadtxt('designs_of_architecture.csv', delimiter=',')
designs_of_components = np.loadtxt('designs_of_components.csv', delimiter=',')

# start the model
model = pyo.ConcreteModel()

# todo read dimitri paper in the introduction

# create the sets of the model using the  pyomo
model.suppliers = pyo.Set(initialize=supplier_list) # index i
model.plants = pyo.Set(initialize=plants_list) # index j
model.retailers = pyo.Set(initialize=retailers_list) # index k
model.collection_centres = pyo.Set(initialize=collection_centres_list) # index m
# model.disassembly_centres = pyo.Set(initialize=disassembly_centres_list) # index d # todo delete everything of dissambly centres
model.remanufacturing_centres = pyo.Set(initialize=remanufacturing_centres_list) # index r # todo change to reproccesing
model.components = pyo.Set(initialize=components_list) # index c # todo change to components
model.periods = pyo.Set(initialize=periods_list) # index p
model.architectures = pyo.Set(initialize=architecture_list) # index a, available architectures
model.r_imperatives = pyo.Set(initialize=r_imperatives_list) #  index e, possible r imperatives
model.design_alternatives = pyo.Set(initialize=designs_list) # index s, possible designs that can be used in parts


# Define the  variables of the model

# continuous variables
model.x = pyo.Var(model.suppliers, model.plants, model.components, model.periods, domain= pyo.NonNegativeReals) # flow from suppliers to plants
model.y = pyo.Var(model.plants,model.retailers, model.periods, domain= pyo.NonNegativeReals) # flow from plants to retailers
model.w = pyo.Var(model.retailers,model.collection_centres, model.periods, domain= pyo.NonNegativeReals) # flow from customers to collection centres
model.a = pyo.Var(model.collection_centres, model.plants, model.periods, domain= pyo.NonNegativeReals) # flow from collection/dissasembly centres to plants
model.b = pyo.Var(model.collection_centres, model.retailers, model.periods, domain= pyo.NonNegativeReals) # flow from collection/dissasembly centres to retailers
model.dk = pyo.Var(model.retailers, model.periods, domain=pyo.NonNegativeReals) # flow from from retailers to disposal
model.dm = pyo.Var(model.collection_centres, model.components, model.periods, domain=pyo.NonNegativeReals) # flow from from collection/dissasembly centres to disposal
model.f = pyo.Var(model.collection_centres, model.remanufacturing_centres, model.components, model.periods, domain=pyo.NonNegativeReals) # flow from collection/dissasembly centres to remanufacturing centres

# divided the e flow into e_rf, e_rm and e_r (refurbishing, remanufacturing and recycling) # todo change the names from refurbishing to repurpose)
model.erf = pyo.Var(model.remanufacturing_centres, model.plants, model.components, model.periods, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centres due to refurbishing
model.erm = pyo.Var(model.remanufacturing_centres, model.plants, model.components, model.periods, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centre due to remanufacturing
model.er = pyo.Var(model.remanufacturing_centres, model.plants, model.components, model.periods, domain=pyo.NonNegativeReals) # flow from disassembly centre to remanufacturing centre due to recycling

# binary variables
model.opm = pyo.Var(model.collection_centres, domain=pyo.Binary) # if we open the collection center m
model.opr = pyo.Var(model.remanufacturing_centres, domain=pyo.Binary) # if we open the reprocessing centre r
model.ops = pyo.Var(model.suppliers, domain=pyo.Binary)  # if we open supplier i



model.ar = pyo.Var(model.architectures, domain= pyo.Binary) # binary, 1 if the product follows architecture a
model.de = pyo.Var(model.design_alternatives, model.components, domain= pyo.Binary) # 1 if the design alternative s is used for part c

model.rimp = pyo.Var(model.r_imperatives, domain=pyo.Binary) # if r imperative e is possible

# variable to define the save the objective function value (just to have a nice code)
model.monetary_costs = pyo.Var(domain=pyo.NonNegativeReals)

# objective function
model.objective = pyo.Objective(expr=model.monetary_costs, sense=pyo.minimize)

# Constraints

model.objective_relationship = pyo.ConstraintList()
model.objective_relationship.add(
    # supplier costs # transporation cost dependant on component and design alternative #todo add design alternative index and add design alternativ index to capacitues of the suppliers
    sum(model.x[i,j,c,p] * flow_cost_suppliers_plants[i,j] for i in model.suppliers for j in model.plants for c in model.components for p in model.periods)
    # +sum(model.x[i,j,c,p] * purchase_cost_suppliers[c,d] for i in model.suppliers for j in model.plants for c in model.parts for p in model.periods for d in model.design_alternatives)
    # plants costs
    + sum(model.y[j,k,p] * flow_cost_plants_retailers[j,k] for j in model.plants for k in model.retailers for p in model.periods)
    # retailers costs
    + sum(model.w[k,m,p] * flow_cost_retailers_collection_centres[k,m] for k in model.retailers for m in model.collection_centres for p in model.periods)
    # collection centres costs #todo a and b are reusing
    + sum(model.a[m,j,p] * flow_cost_collection_centres_plants[m,j] for m in model.collection_centres for j in model.plants for p in model.periods)
    + sum(model.b[m,k,p] * flow_cost_collection_retailer[m,k] for m in model.collection_centres for k in model.retailers for p in model.periods)
    + sum(model.f[m,r,c,p] * flow_cost_collection_centres_remanufacturing[m,r] for m in model.collection_centres for r in model.remanufacturing_centres for c in model.components for p in model.periods)
    # reprocessing centre costs
    + sum(model.erf[r,j,c,p] * flow_cost_remanufacturing_refurbishing[r,j] for r in model.remanufacturing_centres for j in model.plants for c in model.components for p in model.periods)
    + sum(model.er[r,j,c,p] * flow_cost_remanufacturing_reclycling[r,j] for r in model.remanufacturing_centres for j in model.plants for c in model.components for p in model.periods)
    + sum(model.erm[r,j,c,p] * flow_cost_remanufacturing_remanufacturing[r,j] for r in model.remanufacturing_centres for j in model.plants for c in model.components for p in model.periods)
    # todo add flow form reprocessing to retailers (another r imperative, repairing. refurbihsing)
    # opening costs
    + sum(model.opm[m] * opening_cost_collection[m] for m in model.collection_centres)
    + sum(model.opr[r] * opening_cost_reprocessing[r] for r in model.remanufacturing_centres )
    + sum(model.ops[i] * opening_cost_supplier[i] for i in model.suppliers)

    <= model.monetary_costs)


# # constraint 1: capacity of suppliers
model.capacity_suppliers_constraints = pyo.ConstraintList()
for i in model.suppliers:
    for c in model.components:
        for p in model.periods:
            model.capacity_suppliers_constraints.add(sum(model.x[i,j,c,p] for j in model.plants) <= suppliers_capacity[i,c])


# constraint 2: capacity of plants
model.capacity_plants_constraints = pyo.ConstraintList()
for j in model.plants:
    for p in model.periods:
        model.capacity_plants_constraints.add(sum(model.y[j,k,p] for k in model.retailers) <= plants_capacity[j])

# constraint 3: demand of retailers
model.demand_retailers_constraints = pyo.ConstraintList()
for k in model.retailers:
    for p in model.periods:
        model.demand_retailers_constraints.add(sum(model.y[j,k,p] for j in model.plants) >= retailer_demand[k,p])

# constraint 4: capacity of the collection/disassembly centre
model.capacity_collection_centres_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for s in model.design_alternatives:
        for c in model.components:
            for a in model.architectures:
                for p in model.periods:
                    model.capacity_collection_centres_constraints.add((sum(model.a[m, j,p] for j in model.plants)
                                                                  + sum(model.b[m, k,p] for k in model.retailers)) * model.ar[a] * bill_of_materials[a,c]
                                                                      + sum(model.f[m, r, c,p] for r in model.remanufacturing_centres)
                                                                      <= collection_centres_capacity[m, c] * model.opm[m])


# constraint 5: capacity of the remanufacturing centres
model.remanufacturing_centres_capacity_constraints = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for c in model.components:
        for p in model.periods:
            model.remanufacturing_centres_capacity_constraints.add(sum(model.erf[r,j,c,p] for j in model.plants)
                                                               + sum(model.erm[r,j,c,p] for j in model.plants)
                                                               + sum(model.er[r,j,c,p] for j in model.plants)
                                                               <= remanufacturing_centres_capacity[r,c] * model.opr[r])


# constraint 6: flows of the plants
model.plants_flow = pyo.ConstraintList()
for j in model.plants:
    for c in model.components:
        for p in model.periods:
                model.plants_flow.add(sum(model.x[i,j,c,p] for i in model.suppliers)
                                      + sum(model.erf[r, j, c,p] for r in model.remanufacturing_centres)
                                      + sum(model.erm[r, j, c,p] for r in model.remanufacturing_centres)
                                      + sum(model.er[r, j, c,p] for r in model.remanufacturing_centres)
                                      + sum(bill_of_materials[a,c] * model.ar[a] * model.a[m,j,p] for a in model.architectures for m in model.collection_centres)
                                      - sum(bill_of_materials[a,c] * model.ar[a] * model.y[j,k,p] for a in model.architectures for k in model.retailers)
                                      == 0 )
# constraint 7: flow of the retailers
model.retailers_flow = pyo.ConstraintList()
for k in model.retailers:
    for p in model.periods:
        model.retailers_flow.add(sum(model.y[j,k,p] for j in model.plants)
                                 + sum(model.b[m,k,p] for m in model.collection_centres)
                                 - sum(model.w[k,m,p] for m in model.collection_centres)
                                 - model.dk[k,p]
                                 == 0)


# constraint 8: flow of collection/disassembly centres
model.collections_centres_flow = pyo.ConstraintList()
for m in model.collection_centres:
    for c in model.components:
        for p in model.periods:
            model.collections_centres_flow.add(sum(bill_of_materials[a,c] * model.ar[a] * model.w[k,m,p] for k in model.retailers for a in model.architectures)
                                 - sum(bill_of_materials[a,c] * model.ar[a] * model.a[m,j,p] for j in model.plants for a in model.architectures)
                                 - sum(bill_of_materials[a, c] * model.ar[a] * model.b[m, k,p] for j in model.plants for a in model.architectures)
                                 - model.dm[m,c,p]
                                 - sum(model.f[m,r,c,p] for r in model.remanufacturing_centres)
                                 == 0)



# constraint 9: flow of remanufacturing centres
model.remanufacturing_centres_flow = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for c in model.components:
        for p in model.periods:
            model.remanufacturing_centres_flow.add(sum(model.f[m,r,c,p] for m in model.collection_centres)
                                                   - sum(model.erf[r,j,c,p] for j in model.plants)
                                                   - sum(model.erm[r,j,c,p] for j in model.plants)
                                                   - sum(model.er[r,j,c,p] for j in model.plants)
                                                   == 0)

# constraints 10: we have to select one and only one architecture
model.architecture_limits = pyo.ConstraintList()
model.architecture_limits.add(sum(model.ar[a] for a in model.ar) == 1)

# Constraint 11: restrict the refurbishing flow if it is not possible
# restrict the refurbishing flow if the r-imperative is not possible. Refurbishing has the index 0
model.refurbishing_possible = pyo.ConstraintList()

for r in model.remanufacturing_centres:
        for c in model.components:
            for p in model.periods:
                model.refurbishing_possible.add(model.erf[r,j,c,p] <= model.rimp[0]*big_m)

# Constraint 12: restrict the remanufacturing flow if it is not possible
# restrict the remanufacturing flow if the r-imperative is not possible. Remanufacturing has the index 1
model.remanufacturing_possible = pyo.ConstraintList()

for r in model.remanufacturing_centres:
        for c in model.components:
            for p in model.periods:
                model.remanufacturing_possible.add(model.erm[r,j,c,p] <= model.rimp[1]*big_m)

# Constraint 13: restrict the recylcling flow if it is not possible
# restrict the recycling flow if the r-imperative is not possible. Recycling has the index 2
model.recycling_possible = pyo.ConstraintList()

for r in model.remanufacturing_centres:
        for c in model.components:
            for p in model.periods:
                model.recycling_possible.add(model.erf[r,j,c,p] <= model.rimp[2]*big_m)


# Constraint 14: restrict the resusing flow if it is not possible
# restrict the reusing flow if the r-imperative is not possible. Reusing has the index 3
model.reusing_possible = pyo.ConstraintList()
for m in model.collection_centres:
    for k in model.retailers:
        for p in model.periods:
                model.reusing_possible.add(model.b[m,k,p] <= model.rimp[3]*big_m)

# Constraint 15: restrict the repackaging flow if it is not possible
# restrict the repacking flow if the r-imperative is not possible. Repacking has the index 4
model.repacking_possible = pyo.ConstraintList()
for m in model.collection_centres:
    for j in model.plants:
        for p in model.periods:
                model.repacking_possible.add(model.a[m,j,p] <= model.rimp[4]*big_m)

# constraint 16
# restrict the r-imperative given the selection of architecture
model.r_imperative_possible_architecture = pyo.ConstraintList()
for e in model.r_imperatives:
    for a in model.architectures:
        model.r_imperative_possible_architecture.add(model.rimp[e] * model.ar[a] <= r_imperatives_of_architecture[a,e]) #todo linearize this constraint

# constraint 17
# restrict the r-imperative given the selection of design
model.r_imperative_possible_design = pyo.ConstraintList()
for e in model.r_imperatives:
    for s in model.design_alternatives:
        model.r_imperative_possible_design.add(model.rimp[e] * sum(model.de[s,c] for c in model.components) <= r_imperatives_of_designs[s,e]) #todo linearize this constraint

# constraint 18
# relate the part ot a design, only one design can be selected for each part
model.design_to_part = pyo.ConstraintList()
for c in model.components:
    model.design_to_part.add(sum(model.de[s,c] * designs_of_components[c,s] for s in model.design_alternatives) == 1)

# constraint 19
#restrict the flow of the retailer to the collection centre with the collection rate
model.collection_rate_constraints = pyo.ConstraintList()
for k in model.retailers:
    for p in model.periods:
        model.collection_rate_constraints.add(sum(model.w[k,m,p] for m in model.collection_centres) <= nu* sum(model.y[j,k,p] for j in model.plants))
        # model.collection_rate_constraints.add(sum(model.w[k, m] for m in model.collection_centres) <= nu * retailer_demand[k])


# constraint 20
# restrict the flow of collection centres with the rate of reusing (how many products received can be reused)
model.reusing_rate_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for p in model.periods:
        model.reusing_rate_constraints.add(sum(model.a[m,j,p] for j in model.plants) + sum(model.b[m,k,p] for k in model.retailers) <= sigma * sum(model.w[k,m,p] for k in model.retailers))

# constraint 21
# the disposal of items must be at least of a certain rate that describes the items that don't meet the quality standard
model.disposal_rate_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for p in model.periods:
     model.disposal_rate_constraints.add(model.dm[m,c,p] >= lamda * sum(model.w[k,m,p] for k in model.retailers))

# remanufacturing centres r-imperative rates

#constrain 22
model.refurbishing_rate_constraints = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for p in model.periods:
        model.refurbishing_rate_constraints.add(sum(model.erf[r,j, c,p] for j in model.plants for c in model.components) <= alpha * sum(model.f[m,r,c,p] for m in model.collection_centres for c in model.components))

# constrain 23
model.remanufacturing_rate_constraints = pyo.ConstraintList()
for r in model.remanufacturing_centres:
    for p in model.periods:
        model.remanufacturing_rate_constraints.add(sum(model.erf[r,j, c,p] for j in model.plants for c in model.components)
                                                   + sum(model.erm[r,j,c,p] for j in model.plants for c in model.components)
                                                   <= beta * sum(model.f[m,r,c,p] for m in model.collection_centres for c in model.components))


# constraint 24: cannot send to the collection centre if the opening cost is not incurred

model.opening_collection_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for k in model.retailers:
        for p in model.periods:
            model.opening_collection_constraints.add(model.w[k,m,p] <= model.opm[m]*big_m)

# constraint 25: cannot send to the reprocessing centre if the opening cost is not incurred
model.opening_reprocessing_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for r in model.remanufacturing_centres:
        for c in model.components:
            for p in model.periods:
                model.opening_reprocessing_constraints.add(model.f[m,r,c,p] <= model.opr[r]*big_m)

# constraint 26: cannot use supplier i if the fixed cost is not incurred
model.opening_supplier_constraints = pyo.ConstraintList()
for i in model.suppliers:
    for j in model.plants:
        for c in model.components:
            for p in model.periods:
                model.opening_supplier_constraints.add(model.x[i, j, c, p] <= model.ops[i] * big_m)


max_time = 25
solver = 'gurobi'
opt = pyo.SolverFactory(solver)
solution = opt.solve(model)

















# form suppliers to plants
for i in model.suppliers:
    for j in model.plants:
        for c in model.components:
            for p in model.periods:
                # if model.x[i,j,c].value != 0:
                    print("supplier:",i,"plant:",j,"part:",c)
                    print(model.x[i, j, c,p].value)



#
#
# from plants to retailers
for j in model.plants:
    for k in model.retailers:
        for p in model.periods:
                # if model.y[j,k].value != 0:
                    print("plant:",j,"retailer:",k)
                    print(model.y[j,k,p].value)




print(model.dk[0,0].value)






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







