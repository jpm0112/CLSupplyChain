import pyomo.environ as pyo
import numpy as np
import pandas as pd



# define the size of each set

# watch out when defining a value to one. The saving fucntion of the csv file may save a list rather than matrix.
# comment the loading section if you want to run a test instance defining one of the number values to one.
supplier_number = 12
plants_number = 1
retailers_number = 10
collection_centres_number = 3
reprocessing_centres_number = 7
components_number = 5
architecture_number = 2
r_imperatives_names = ['repurposing', 'remanufacturing', 'recycling', 'reusing', 'repairing'] # assuming A and B are reusing
r_imperatives_number = len(r_imperatives_names)
designs_number = 9 # for each part, two designs

big_m = 100000 # todo find a good big m

delimiter = "," # ADDED THIS TO CHANGE EASILY THE FORMAT OF THE CSV FILES. JSUT IN CASE YOU AIT MAKE EASIER THE INSTANCE CREATION



supplier_list = list(range(supplier_number))
plants_list = list(range(plants_number))
retailers_list = list(range(retailers_number))
collection_centres_list = list(range(collection_centres_number))
reprocessing_centres_list = list(range(reprocessing_centres_number))
components_list = list(range(components_number))
architecture_list = list(range(architecture_number))
r_imperatives_list = list(range(r_imperatives_number))
designs_list = list(range(designs_number))



# TO GENERATE CSVs, COMMENT THIS SECTION IF YOU WANT TO USE AN ISNTANCE FROM FILES
# parameters
np.random.seed(1048596)
# Initialize capacities with random values within a sensible range
# suppliers_capacity = np.random.randint(400, 500, (supplier_number, designs_number))
# np.savetxt('suppliers_capacity.csv', suppliers_capacity, delimiter=delimiter)
# plants_capacity = np.random.randint(500, 800, (plants_number))
# np.savetxt('plants_capacity.csv', plants_capacity, delimiter=delimiter)
# retailer_demand = np.random.randint(200, 300, (retailers_number))
# np.savetxt('retailer_demand.csv', retailer_demand, delimiter=delimiter)
# collection_centres_capacity = np.random.randint(300, 500, (collection_centres_number, components_number))
# np.savetxt('collection_centres_capacity.csv', collection_centres_capacity, delimiter=delimiter)


# reprocessing_centres_capacity = np.random.randint(300, 400, (reprocessing_centres_number, components_number))
# np.savetxt('reprocessing_centres_capacity.csv', reprocessing_centres_capacity, delimiter=delimiter)

# reprocessing_centres_capacity_repurposing = np.random.randint(300, 400, (reprocessing_centres_number, components_number))
# np.savetxt('reprocessing_centres_capacity_repurposing.csv', reprocessing_centres_capacity_repurposing, delimiter=delimiter)
# reprocessing_centres_capacity_remanufacturing = np.random.randint(300, 400,(reprocessing_centres_number, components_number))
# np.savetxt('reprocessing_centres_capacity_remanufacturing.csv', reprocessing_centres_capacity_remanufacturing, delimiter=delimiter)
# reprocessing_centres_capacity_recycling = np.random.randint(300, 400,(reprocessing_centres_number, components_number))
# np.savetxt('reprocessing_centres_capacity_recycling.csv', reprocessing_centres_capacity_recycling, delimiter=delimiter)
# reprocessing_centres_capacity_repairing = np.random.randint(300, 400, (reprocessing_centres_number))
# np.savetxt('reprocessing_centres_capacity_repairing.csv', reprocessing_centres_capacity_repairing, delimiter=delimiter)



flow_cost_suppliers_plants = np.random.randint(1, 5,(supplier_number, plants_number))  # suppliers as rows, plants as columns
np.savetxt('flow_cost_suppliers_plants.csv', flow_cost_suppliers_plants, delimiter=delimiter)
purchase_cost_suppliers = np.random.randint(1, 5, (components_number, designs_number))  # suppliers as rows, plants as columns
np.savetxt('purchase_cost_suppliers.csv', purchase_cost_suppliers, delimiter=delimiter)
flow_cost_plants_retailers = np.random.randint(1, 5, (plants_number, retailers_number))
np.savetxt('flow_cost_plants_retailers.csv', flow_cost_plants_retailers, delimiter=delimiter)
flow_cost_retailers_collection_centres = np.random.randint(1, 5, (retailers_number, collection_centres_number))
np.savetxt('flow_cost_retailers_collection_centres.csv', flow_cost_retailers_collection_centres, delimiter=delimiter)
flow_cost_collection_retailer = np.random.randint(1, 5, (collection_centres_number, retailers_number))
np.savetxt('flow_cost_collection_retailer.csv', flow_cost_collection_retailer, delimiter=delimiter)
flow_cost_collection_disposal = np.random.randint(1, 5, (collection_centres_number, components_number))
np.savetxt('flow_cost_collection_disposal.csv', flow_cost_collection_disposal, delimiter=delimiter)

flow_cost_collection_centres_reprocessing = np.random.randint(1, 5, (collection_centres_number, reprocessing_centres_number))
np.savetxt('flow_cost_collection_centres_reprocessing.csv', flow_cost_collection_centres_reprocessing, delimiter=delimiter)
flow_cost_collection_centres_reprocessing_products = np.random.randint(1, 5,(collection_centres_number, reprocessing_centres_number))
np.savetxt('flow_cost_collection_centres_reprocessing_products.csv', flow_cost_collection_centres_reprocessing_products, delimiter=delimiter)

flow_cost_collection_centres_plants = np.random.randint(1, 5, (collection_centres_number, plants_number))
np.savetxt('flow_cost_collection_centres_plants.csv', flow_cost_collection_centres_plants, delimiter=delimiter)
flow_cost_disassembly_disposal = np.random.randint(1, 5, (collection_centres_number))
np.savetxt('flow_cost_disassembly_disposal.csv', flow_cost_disassembly_disposal, delimiter=delimiter)

flow_cost_reprocessing_repurposing = np.random.randint(1, 5, (reprocessing_centres_number, plants_number))
np.savetxt('flow_cost_reprocessing_repurposing.csv', flow_cost_reprocessing_repurposing, delimiter=delimiter)
flow_cost_reprocessing_reclycling = np.random.randint(1, 5, (reprocessing_centres_number, plants_number))
np.savetxt('flow_cost_reprocessing_reclycling.csv', flow_cost_reprocessing_reclycling, delimiter=delimiter)
flow_cost_reprocessing_remanufacturing = np.random.randint(1, 5, (reprocessing_centres_number, plants_number))
np.savetxt('flow_cost_reprocessing_remanufacturing.csv', flow_cost_reprocessing_remanufacturing, delimiter=delimiter)
flow_cost_reprocessing_repairing = np.random.randint(1, 5, (reprocessing_centres_number, retailers_number))
np.savetxt('flow_cost_reprocessing_repairing.csv', flow_cost_reprocessing_repairing, delimiter=delimiter)





# opening_cost_collection = np.random.randint(1, 100, (collection_centres_number))
# np.savetxt('opening_cost_collection.csv', opening_cost_collection, delimiter=delimiter)
# opening_cost_reprocessing = np.random.randint(1, 100, (reprocessing_centres_number))
# np.savetxt('opening_cost_reprocessing.csv', opening_cost_reprocessing, delimiter=delimiter)
# opening_cost_supplier = np.random.randint(1, 100, (supplier_number))
# np.savetxt('opening_cost_supplier.csv', opening_cost_supplier, delimiter=delimiter)
opening_cost_plants = np.random.randint(1, 100, (plants_number))
np.savetxt('opening_cost_plants.csv', opening_cost_plants, delimiter=delimiter)

# Initialize other parameters with random or specified values
# bill_of_materials = np.random.randint(1, 2, (architecture_number, components_number))  # rows are the architectures and columns are the part, replaces the r parameter in the model
# np.savetxt('bill_of_materials.csv', bill_of_materials, delimiter=delimiter)
# r_imperatives_of_architecture = np.random.randint(0, 2, (architecture_number,r_imperatives_number))  # rows are architectures and columns the part. It has a value of 1 if the r-imperative is possible with the architecutre
# np.savetxt('r_imperatives_of_architecture.csv', r_imperatives_of_architecture, delimiter=delimiter)
# r_imperatives_of_designs = np.random.randint(0, 2, (designs_number, r_imperatives_number))  # rows are designs and columns the part. It has a value of 1 if the r-imperative is possible with the design
# np.savetxt('r_imperatives_of_designs.csv', r_imperatives_of_designs, delimiter=delimiter)
# designs_of_architecture = np.random.randint(0, 2, (architecture_number,designs_number))  # rows are architecture and columns the design. It has a value of 1 if the design is possible with the architecture
# np.savetxt('designs_of_architecture.csv', designs_of_architecture, delimiter=delimiter)
# designs_of_components = np.random.randint(1, 2, (components_number, designs_number))  # rows are components and columns the design. It has a value of 1 if the design is possible with the part
# np.savetxt('designs_of_components.csv', designs_of_components, delimiter=delimiter)


nu = 0.5  # percentage collected from the retailers
# rates for the outgoing flows of the collection centres
sigma = 0.1  # percentage from the collected product that is good enough for reusing or repackaging
lamda = np.full(designs_number, 0.1)# percentage of the collected components that has to be disposed (depending on the design alternative)
np.savetxt('lamda.csv', lamda, delimiter=delimiter)
gamma = 0.3  # percentage from the collected product that is good enough for repairing
# rates for the reprocessing centres
alpha = np.full(designs_number, 0.1) # % of components received in a reprocessing centre that are good enough to be repurposed (depending on the design alternative)
np.savetxt('alpha.csv', alpha, delimiter=delimiter)
beta = np.full(designs_number, 0.1)  # % of components received in a reprocessing centre that are good enough to be remaufactured (depending on the design alternative)
np.savetxt('beta.csv', beta, delimiter=delimiter)




# TO LOAD CSVs
suppliers_capacity = np.loadtxt('suppliers_capacity.csv', delimiter=delimiter, ndmin=1)
plants_capacity = np.loadtxt('plants_capacity.csv', delimiter=delimiter, ndmin=1)
retailer_demand = np.loadtxt('retailer_demand.csv', delimiter=delimiter, ndmin=1)
collection_centres_capacity = np.loadtxt('collection_centres_capacity.csv', delimiter=delimiter, ndmin=1)
reprocessing_centres_capacity_repurposing = np.loadtxt('reprocessing_centres_capacity_repurposing.csv', delimiter=delimiter, ndmin=2)
reprocessing_centres_capacity_remanufacturing = np.loadtxt('reprocessing_centres_capacity_remanufacturing.csv',delimiter=delimiter, ndmin=2)
reprocessing_centres_capacity_recycling = np.loadtxt('reprocessing_centres_capacity_recycling.csv',delimiter=delimiter, ndmin=2)
reprocessing_centres_capacity_repairing = np.loadtxt('reprocessing_centres_capacity_repairing.csv', delimiter=delimiter,ndmin=2)


# reprocessing_centres_capacity = np.loadtxt('reprocessing_centres_capacity.csv', delimiter=delimiter)

flow_cost_reprocessing_repurposing = np.loadtxt('flow_cost_reprocessing_repurposing.csv', delimiter=delimiter, ndmin=2)
flow_cost_reprocessing_reclycling = np.loadtxt('flow_cost_reprocessing_reclycling.csv', delimiter=delimiter, ndmin=2)
flow_cost_reprocessing_remanufacturing = np.loadtxt('flow_cost_reprocessing_remanufacturing.csv', delimiter=delimiter,ndmin=2)
flow_cost_reprocessing_repairing = np.loadtxt('flow_cost_reprocessing_repairing.csv', delimiter=delimiter, ndmin=2)

flow_cost_suppliers_plants = np.loadtxt('flow_cost_suppliers_plants.csv', delimiter=delimiter, ndmin=2)
purchase_cost_suppliers = np.loadtxt('purchase_cost_suppliers.csv', delimiter=delimiter, ndmin=2)           *100000
flow_cost_plants_retailers = np.loadtxt('flow_cost_plants_retailers.csv', delimiter=delimiter, ndmin=2)
flow_cost_retailers_collection_centres = np.loadtxt('flow_cost_retailers_collection_centres.csv', delimiter=delimiter,ndmin=2)
flow_cost_collection_retailer = np.loadtxt('flow_cost_collection_retailer.csv', delimiter=delimiter, ndmin=2)
flow_cost_collection_disposal = np.loadtxt('flow_cost_collection_disposal.csv', delimiter=delimiter, ndmin=2)

flow_cost_collection_centres_reprocessing = np.loadtxt('flow_cost_collection_centres_reprocessing.csv', delimiter=delimiter,ndmin=2)
flow_cost_collection_centres_reprocessing_products = np.loadtxt('flow_cost_collection_centres_reprocessing_products.csv',delimiter=delimiter,ndmin=2)
flow_cost_collection_centres_plants = np.loadtxt('flow_cost_collection_centres_plants.csv', delimiter=delimiter, ndmin=2)
flow_cost_disassembly_disposal = np.loadtxt('flow_cost_disassembly_disposal.csv', delimiter=delimiter, ndmin=1)
flow_cost_reprocessing_repurposing = np.loadtxt('flow_cost_reprocessing_repurposing.csv', delimiter=delimiter, ndmin=2)
flow_cost_reprocessing_reclycling = np.loadtxt('flow_cost_reprocessing_reclycling.csv', delimiter=delimiter, ndmin=2)
flow_cost_reprocessing_remanufacturing = np.loadtxt('flow_cost_reprocessing_remanufacturing.csv', delimiter=delimiter,ndmin=2)




opening_cost_collection = np.loadtxt('opening_cost_collection.csv', delimiter=delimiter, ndmin=1) /1000000
opening_cost_reprocessing = np.loadtxt('opening_cost_reprocessing.csv', delimiter=delimiter, ndmin=1) /100000
opening_cost_supplier = np.loadtxt('opening_cost_supplier.csv', delimiter=delimiter, ndmin=1)
opening_cost_plants = np.loadtxt('opening_cost_plants.csv', delimiter=delimiter, ndmin=1)

bill_of_materials = np.loadtxt('bill_of_materials.csv', delimiter=delimiter, ndmin=1)

r_imperatives_of_architecture = np.loadtxt('r_imperatives_of_architecture.csv', delimiter=delimiter, ndmin=1)
r_imperatives_of_designs = np.loadtxt('r_imperatives_of_designs.csv', delimiter=delimiter, ndmin=1)

designs_of_architecture = np.loadtxt('designs_of_architecture.csv', delimiter=delimiter, ndmin=1)
designs_of_components = np.loadtxt('designs_of_components.csv', delimiter=delimiter, ndmin=1)

flow_cost_reprocessing_repairing = np.loadtxt('flow_cost_reprocessing_repairing.csv', delimiter=delimiter, ndmin=1)

lamda = np.loadtxt('lamda.csv', delimiter=delimiter, ndmin=1)
alpha = np.loadtxt('alpha.csv', delimiter=delimiter, ndmin=1)
beta = np.loadtxt('beta.csv', delimiter=delimiter, ndmin=1)




# start the model
model = pyo.ConcreteModel()

# create the sets of the model using the  pyomo
model.suppliers = pyo.Set(initialize=supplier_list) # index i
model.plants = pyo.Set(initialize=plants_list) # index j
model.retailers = pyo.Set(initialize=retailers_list) # index k
model.collection_centres = pyo.Set(initialize=collection_centres_list) # index m
model.reprocessing_centres = pyo.Set(initialize=reprocessing_centres_list) # index r
model.components = pyo.Set(initialize=components_list) # index c
model.architectures = pyo.Set(initialize=architecture_list) # index a, available architectures
model.r_imperatives = pyo.Set(initialize=r_imperatives_list) #  index e, possible r imperatives
model.design_alternatives = pyo.Set(initialize=designs_list) # index s, possible designs that can be used in components


# Define the  variables of the model

# continuous variables
model.x = pyo.Var(model.suppliers, model.plants, model.components, model.design_alternatives, domain= pyo.NonNegativeReals) # flow from suppliers to plants
model.y = pyo.Var(model.plants,model.retailers,  domain= pyo.NonNegativeReals) # flow from plants to retailers
model.w = pyo.Var(model.retailers,model.collection_centres,  domain= pyo.NonNegativeReals) # flow from customers to collection centres
model.a = pyo.Var(model.collection_centres, model.plants,  domain= pyo.NonNegativeReals) # flow from collection/dissasembly centres to plants
model.b = pyo.Var(model.collection_centres, model.retailers,  domain= pyo.NonNegativeReals) # flow from collection/dissasembly centres to retailers
model.dk = pyo.Var(model.retailers, domain=pyo.NonNegativeReals) # flow from retailers to disposal
model.dm = pyo.Var(model.collection_centres, model.components,  domain=pyo.NonNegativeReals) # flow from from collection/dissasembly centres to disposal
model.f = pyo.Var(model.collection_centres, model.reprocessing_centres, model.components,  domain=pyo.NonNegativeReals) # flow from collection/dissasembly centres to remanufacturing centres (components)
model.g = pyo.Var(model.collection_centres, model.reprocessing_centres,domain=pyo.NonNegativeReals)  # flow from collection/dissasembly centres to remanufacturing centres (products)

# divided the e flow into e_rf, e_rm and e_r (repurposing, remanufacturing and recycling)
model.erp = pyo.Var(model.reprocessing_centres, model.plants, model.components,  domain=pyo.NonNegativeReals) # flow from reprocessing centre to plants centres due to repurposing
model.erm = pyo.Var(model.reprocessing_centres, model.plants, model.components,  domain=pyo.NonNegativeReals) # flow from reprocessing centre to plants centre due to remanufacturing
model.er = pyo.Var(model.reprocessing_centres, model.plants, model.components, domain=pyo.NonNegativeReals) # flow from reprocessing centre to plants centre due to recycling
model.c = pyo.Var(model.reprocessing_centres, model.retailers, domain=pyo.NonNegativeReals)  # flow from reprocessing centre to retailers centres due to repairing

# binary variables
model.opm = pyo.Var(model.collection_centres, domain=pyo.Binary) # if we open the collection center m
model.opr = pyo.Var(model.reprocessing_centres, domain=pyo.Binary) # if we open the reprocessing centre r
model.ops = pyo.Var(model.suppliers, domain=pyo.Binary)  # if we open supplier i
model.opp = pyo.Var(model.plants, domain=pyo.Binary)  # if we open plant j


model.ar = pyo.Var(model.architectures, domain= pyo.Binary) # binary, 1 if the product follows architecture a
model.de = pyo.Var(model.design_alternatives, model.components, domain= pyo.Binary) # 1 if the design alternative s is used for part c

model.rimp = pyo.Var(model.r_imperatives, domain=pyo.Binary) # if r imperative e is possible

# variable to define the save the objective function value (just to have a nice code)
model.monetary_costs = pyo.Var(domain=pyo.NonNegativeReals)

# number of the components that come from the reverse flow (either to the plants or directly to the retailers)
model.number_reutilized_components = pyo.Var( domain=pyo.NonNegativeReals)

# objective function
# model.objective = pyo.Objective(expr=model.monetary_costs, sense=pyo.minimize)
model.objective = pyo.Objective(expr=model.number_reutilized_components, sense=pyo.maximize)


# auxiliary variable to linearize the model
model.aux_ar_de = pyo.Var(model.architectures,model.design_alternatives,model.components, domain=pyo.Binary)  # multiplication between ar and de varibales




# Constraints

model.objective_relationship_costs = pyo.ConstraintList()
model.objective_relationship_costs.add(
    # supplier costs # transporation cost dependant on component and design alternative
    sum(model.x[i,j,c,d] * flow_cost_suppliers_plants[i,j] for i in model.suppliers for j in model.plants for c in model.components for d in model.design_alternatives)
    + sum(model.x[i,j,c,d] * purchase_cost_suppliers[c,d] for i in model.suppliers for j in model.plants for c in model.components for d in model.design_alternatives)
    # plants costs
    + sum(model.y[j,k] * flow_cost_plants_retailers[j,k] for j in model.plants for k in model.retailers )
    # retailers costs
    + sum(model.w[k,m] * flow_cost_retailers_collection_centres[k,m] for k in model.retailers for m in model.collection_centres )
    # collection centres costs # a and b are reusing
    + sum(model.a[m,j] * flow_cost_collection_centres_plants[m,j] for m in model.collection_centres for j in model.plants )
    + sum(model.b[m,k] * flow_cost_collection_retailer[m,k] for m in model.collection_centres for k in model.retailers )
    + sum(model.f[m,r,c] * flow_cost_collection_centres_reprocessing[m,r] for m in model.collection_centres for r in model.reprocessing_centres for c in model.components )
    + sum(model.g[m, r] * flow_cost_collection_centres_reprocessing_products[m, r] for m in model.collection_centres for r in model.reprocessing_centres)
    + sum(model.dm[m, c] * flow_cost_collection_disposal[m, c] for m in model.collection_centres for c in model.components)

    # reprocessing centre costs
    + sum(model.erp[r,j,c] * flow_cost_reprocessing_repurposing[r,j] for r in model.reprocessing_centres for j in model.plants for c in model.components )
    + sum(model.er[r,j,c] * flow_cost_reprocessing_reclycling[r,j] for r in model.reprocessing_centres for j in model.plants for c in model.components )
    + sum(model.erm[r,j,c] * flow_cost_reprocessing_remanufacturing[r,j] for r in model.reprocessing_centres for j in model.plants for c in model.components )
    + sum(model.c[r,k] * flow_cost_reprocessing_repairing[r, k] for r in model.reprocessing_centres for k in model.retailers)

    # opening costs
    + sum(model.opm[m] * opening_cost_collection[m] for m in model.collection_centres)
    + sum(model.opr[r] * opening_cost_reprocessing[r] for r in model.reprocessing_centres)
    + sum(model.ops[i] * opening_cost_supplier[i] for i in model.suppliers)
    + sum(model.opp[j] * opening_cost_plants[j] for j in model.plants)

    <= model.monetary_costs)

model.objective_relationship_reutilization = pyo.ConstraintList()
model.objective_relationship_reutilization.add(
    sum(model.erp[r,j,c]+model.erm[r,j,c]+model.er[r,j,c] for r in model.reprocessing_centres for j in model.plants for c in model.components)
    + sum(model.c[r,k]*bill_of_materials[a,c]*model.ar[a] for r in model.reprocessing_centres for k in model.retailers for c in model.components for a in model.architectures)
    + sum(model.a[m,j]*bill_of_materials[a,c]*model.ar[a] for m in model.collection_centres for j in model.plants for c in model.components for a in model.architectures)
    + sum(model.b[m, k] * bill_of_materials[a, c] * model.ar[a] for m in model.collection_centres for k in model.retailers for c in model.components for a in model.architectures)
    >= model.number_reutilized_components)


#CAPACITY CONSTRAINTS

# # constraint 1: capacity of suppliers
model.capacity_suppliers_constraints = pyo.ConstraintList()
for i in model.suppliers:
        for d in model.design_alternatives:
            model.capacity_suppliers_constraints.add(sum(model.x[i,j,c,d] for j in model.plants for c in model.components) <= suppliers_capacity[i,d])


# constraint 2: capacity of plants
model.capacity_plants_constraints = pyo.ConstraintList()
for j in model.plants:
        model.capacity_plants_constraints.add(sum(model.y[j,k] for k in model.retailers) <= plants_capacity[j])

# constraint 3: demand of retailers
model.demand_retailers_constraints = pyo.ConstraintList()
for k in model.retailers:
        model.demand_retailers_constraints.add(sum(model.y[j,k] for j in model.plants) >= retailer_demand[k])

# constraint 4: capacity of the collection/disassembly centre
model.capacity_collection_centres_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for s in model.design_alternatives:
        for c in model.components:
            for a in model.architectures:

                    model.capacity_collection_centres_constraints.add((sum(model.a[m, j] for j in model.plants)
                                                                  + sum(model.b[m, k] for k in model.retailers)) * model.ar[a] * bill_of_materials[a,c]
                                                                      + sum(model.f[m, r, c] for r in model.reprocessing_centres)
                                                                      <= collection_centres_capacity[m, c] * model.opm[m])


# constraint 5: capacity of the reprocessing centres
model.reprocessing_centres_capacity_constraints = pyo.ConstraintList()
for r in model.reprocessing_centres:
    for c in model.components:

            # model.reprocessing_centres_capacity_constraints.add(sum(model.erp[r,j,c] for j in model.plants)
            #                                                     + sum(model.erm[r,j,c] for j in model.plants)
            #                                                     + sum(model.er[r,j,c] for j in model.plants)
            #                                                     + sum(model.c[r, k] * bill_of_materials[a, c] for k in model.retailers for a in model.architectures)
            #                                                     <= reprocessing_centres_capacity[r,c] * model.opr[r])

            model.reprocessing_centres_capacity_constraints.add(sum(model.erp[r, j, c] for j in model.plants)
                                                                <= reprocessing_centres_capacity_repurposing[r, c] * model.opr[r])
            model.reprocessing_centres_capacity_constraints.add(sum(model.erm[r, j, c] for j in model.plants)
                                                                <= reprocessing_centres_capacity_remanufacturing[r, c] * model.opr[r])
            model.reprocessing_centres_capacity_constraints.add(sum(model.er[r, j, c] for j in model.plants)
                                                                <= reprocessing_centres_capacity_recycling[r, c] * model.opr[r])
            model.reprocessing_centres_capacity_constraints.add(sum(model.c[r, k]  for k in model.retailers)
                                                                <= reprocessing_centres_capacity_repairing[r] * model.opr[r])


# FLOWS CONSTRAINTS

# constraint 6: flows of the plants
model.plants_flow = pyo.ConstraintList()
for j in model.plants:
    for c in model.components:

                model.plants_flow.add(sum(model.x[i,j,c,d] for i in model.suppliers for d in model.design_alternatives)
                                      + sum(model.erp[r, j, c] for r in model.reprocessing_centres)
                                      + sum(model.erm[r, j, c] for r in model.reprocessing_centres)
                                      + sum(model.er[r, j, c] for r in model.reprocessing_centres)
                                      + sum(bill_of_materials[a,c] * model.ar[a] * model.a[m,j] for a in model.architectures for m in model.collection_centres)
                                      - sum(bill_of_materials[a,c] * model.ar[a] * model.y[j,k] for a in model.architectures for k in model.retailers)
                                      == 0 )
# constraint 7: flow of the retailers
model.retailers_flow = pyo.ConstraintList()
for k in model.retailers:

        model.retailers_flow.add(sum(model.y[j,k] for j in model.plants)
                                 + sum(model.b[m,k] for m in model.collection_centres)
                                 - sum(model.w[k,m] for m in model.collection_centres)
                                 - model.dk[k]
                                 == 0)


# constraint 8: flow of collection/disassembly centres
model.collections_centres_flow = pyo.ConstraintList()
for m in model.collection_centres:
    for c in model.components:

            model.collections_centres_flow.add(sum(bill_of_materials[a,c] * model.ar[a] * model.w[k,m] for k in model.retailers for a in model.architectures)
                                 - sum(bill_of_materials[a,c] * model.ar[a] * model.a[m,j] for j in model.plants for a in model.architectures)
                                 - sum(bill_of_materials[a, c] * model.ar[a] * model.b[m, k] for k in model.retailers for a in model.architectures)
                                 - model.dm[m,c]
                                 - sum(model.f[m,r,c] for r in model.reprocessing_centres)
                                 == 0)



# constraint 9: flow of reprocessing centres
model.reprocessing_centres_flow = pyo.ConstraintList()
for r in model.reprocessing_centres:
    for c in model.components:

            model.reprocessing_centres_flow.add(sum(model.f[m,r,c] for m in model.collection_centres)
                                                - sum(model.erp[r,j,c] for j in model.plants)
                                                - sum(model.erm[r,j,c] for j in model.plants)
                                                - sum(model.er[r,j,c] for j in model.plants)
                                                == 0)

for r in model.reprocessing_centres:
    model.reprocessing_centres_flow.add(sum(model.c[r, k] for k in model.retailers) == sum(model.g[m, r] for m in model.collection_centres))







# restrict the flows with the rimp variable that describes if a certain architecture or desing alternative allows the r imperative

# Constraint 11: restrict the repurpose flow if it is not possible
# restrict the repurposing flow if the r-imperative is not possible. Repurposing has the index 0
model.repurposing_possible = pyo.ConstraintList()

for r in model.reprocessing_centres:
        for c in model.components:
            for j in model.plants:
                model.repurposing_possible.add(model.erp[r,j,c] <= model.rimp[0] * big_m)

# Constraint 12: restrict the remanufacturing flow if it is not possible
# restrict the remanufacturing flow if the r-imperative is not possible. Remanufacturing has the index 1
model.remanufacturing_possible = pyo.ConstraintList()

for r in model.reprocessing_centres:
        for c in model.components:
            for j in model.plants:
                model.remanufacturing_possible.add(model.erm[r,j,c] <= model.rimp[1]*big_m)

# Constraint 13: restrict the recylcling flow if it is not possible
# restrict the recycling flow if the r-imperative is not possible. Recycling has the index 2
model.recycling_possible = pyo.ConstraintList()

for r in model.reprocessing_centres:
        for c in model.components:
            for j in model.plants:
                model.recycling_possible.add(model.er[r,j,c] <= model.rimp[2] * big_m)


# Constraint 14: restrict the resusing flow if it is not possible
# restrict the reusing flow if the r-imperative is not possible. Reusing has the index 3
model.reusing_possible = pyo.ConstraintList()
for m in model.collection_centres:
    for k in model.retailers:
                model.reusing_possible.add(model.b[m,k] <= model.rimp[3]*big_m)

# Constraint 15: restrict the repackaging flow if it is not possible
# restrict the repacking flow if the r-imperative is not possible. Reusing has the index 3
model.repacking_possible = pyo.ConstraintList()
for m in model.collection_centres:
    for j in model.plants:
                model.repacking_possible.add(model.a[m,j] <= model.rimp[3]*big_m)

# Constraint 15: restrict the repackaging flow if it is not possible
# restrict the repacking flow if the r-imperative is not possible. repairing has the index 4
model.repairing_possible = pyo.ConstraintList()
for r in model.reprocessing_centres:
    for k in model.retailers:
        model.repairing_possible.add(model.c[r, k] <= model.rimp[4] * big_m)






# constraints 10: we have to select one and only one architecture
model.architecture_limits = pyo.ConstraintList()
model.architecture_limits.add(sum(model.ar[a] for a in model.ar) == 1)

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



# RATE CONSTRAINTS
# constraint 19
#restrict the flow of the retailer to the collection centre with the collection rate
model.collection_rate_constraints = pyo.ConstraintList()
for k in model.retailers:
        model.collection_rate_constraints.add(sum(model.w[k,m] for m in model.collection_centres) <= nu * sum(model.y[j,k] for j in model.plants))


# constraint 20
# restrict the flow of collection centres with the rate of reusing (how many products received can be reused)
model.reusing_rate_constraints = pyo.ConstraintList()
for m in model.collection_centres:
        model.reusing_rate_constraints.add(sum(model.a[m,j] for j in model.plants) + sum(model.b[m,k] for k in model.retailers) <= sigma * sum(model.w[k,m] for k in model.retailers))

# restrict the flow of collection centres with the rate of repairing (how many products received can be used after repairing)
model.repairing_rate_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    model.reusing_rate_constraints.add(
        sum(model.a[m, j] for j in model.plants) + sum(model.b[m, k] for k in model.retailers)
        + sum(model.g[m,r] for r in model.reprocessing_centres)
        <= gamma * sum(model.w[k, m] for k in model.retailers))

# constraint 21
# the disposal of items must be at least of a certain rate that describes the items that don't meet the quality standard
model.disposal_rate_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for c in model.components:
        model.disposal_rate_constraints.add(model.dm[m,c] >= sum( model.w[k,m] * bill_of_materials[a,c]  * lamda[s] * model.aux_ar_de[a,s,c] for k in model.retailers for a in model.architectures for s in model.design_alternatives))



# reprocessing centres r-imperative rates
#constraint 22:
model.repurposing_rate_constraints = pyo.ConstraintList()
for r in model.reprocessing_centres:
        model.repurposing_rate_constraints.add(sum(model.erp[r,j, c] for j in model.plants for c in model.components)  <= sum(model.f[m,r,c] * alpha[s] * model.de[s,c] for s in model.design_alternatives for m in model.collection_centres for c in model.components))

# constrain 23
model.reprocessing_rate_constraints = pyo.ConstraintList()
for r in model.reprocessing_centres:
        model.reprocessing_rate_constraints.add(sum(model.erp[r,j, c] for j in model.plants for c in model.components)
                                                + sum(model.erm[r,j,c] for j in model.plants for c in model.components)
                                                <=  sum(model.f[m,r,c] * beta[s] * model.de[s,c] for s in model.design_alternatives for m in model.collection_centres for c in model.components))


# constraint 24: cannot send to the collection centre if the opening cost is not incurred
model.opening_collection_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for k in model.retailers:
            model.opening_collection_constraints.add(model.w[k,m] <= model.opm[m]*big_m)

# constraint 25: cannot send to the reprocessing centre if the opening cost is not incurred
model.opening_reprocessing_constraints = pyo.ConstraintList()
for m in model.collection_centres:
    for r in model.reprocessing_centres:
        for c in model.components:
                model.opening_reprocessing_constraints.add(model.f[m,r,c] <= model.opr[r]*big_m)

# constraint 26: cannot use supplier i if the fixed cost is not incurred
model.opening_supplier_constraints = pyo.ConstraintList()
for i in model.suppliers:
    for j in model.plants:
        for c in model.components:
            for d in model.design_alternatives:
                model.opening_supplier_constraints.add(model.x[i, j, c,d] <= model.ops[i] * big_m)

# constraint 27: cannot use plant j if the fixed cost is not incurred
model.opening_plants_constraints = pyo.ConstraintList()
for j in model.plants:
    for k in model.retailers:
            model.opening_plants_constraints.add(model.y[j,k] <= model.opp[j] * big_m)





model.auxiliary_contraints_ar_de = pyo.ConstraintList()
for a in model.architectures:
    for s in model.design_alternatives:
        for c in model.components:
            model.auxiliary_contraints_ar_de.add(model.aux_ar_de[a, s,c] <= model.ar[a])
            model.auxiliary_contraints_ar_de.add(model.aux_ar_de[a, s,c] <= model.de[s,c])
            model.auxiliary_contraints_ar_de.add(model.aux_ar_de[a, s,c] >= model.ar[a] + model.de[s,c] - 1)











max_time = 25
solver = 'gurobi'
opt = pyo.SolverFactory(solver)
solution = opt.solve(model)

for a in model.architectures:
    # if model.ar[a].value != 0:
    print("arch:", a)
    print(model.ar[a].value)

for s in model.design_alternatives:
    for c in model.components:
        if model.de[s, c].value != 0:
            print("design alternative:", s, "component:", c)
            print(model.de[s, c].value)

# form suppliers to plants
print("Variable X")
for i in model.suppliers:
    for j in model.plants:
        for c in model.components:
            for d in model.design_alternatives:
                if model.x[i,j,c,d].value != 0:
                    print("supplier:",i,"plant:",j,"component:",c, "design: ", d)
                    print(model.x[i, j, c,d].value)

print("\nVariable Y")
for j in model.plants:
    for k in model.retailers:
        if model.y[j, k].value != 0:
            print("plants:", j, "retailers:", k)
            print(model.y[j, k].value)


print("\nVariable W")
for k in model.retailers:
    for m in model.collection_centres:
        if model.w[k, m].value != 0:
            print("retailers:", k, "collection_centres:", m)
            print(model.w[k, m].value)
#
#
# print("\nVariable A")
# for m in model.collection_centres:
#     for j in model.plants:
#         if model.a[m, j].value != 0:
#             print("collection_centres:", m, "plants:", j)
#             print(model.a[m, j].value)
#
#
# print("\nVariable B")
# for m in model.collection_centres:
#     for k in model.retailers:
#         if model.b[m, k].value != 0:
#             print("collection_centres:", m, "retailers:", k)
#             print(model.b[m, k].value)
#
#
# print("\nVariable DK")
# for k in model.retailers:
#         if model.dk[k].value != 0:
#             print("retailers:", k)
#             print(model.dk[k].value)
#
#
# print("\nVariable DM")
# for m in model.collection_centres:
#     for c in model.components:
#         if model.dm[m, c].value != 0:
#             print("collection_centres:", m, "components:", c)
#             print(model.dm[m, c].value)
#
#
print("\nVariable F")
for m in model.collection_centres:
    for r in model.reprocessing_centres:
        for c in model.components:
            if model.f[m, r, c].value != 0:
                print("collection_centres:", m, "reprocessing_centres:", r, "components:", c)
                print(model.f[m, r, c].value)

#
# print("\nVariable G")
# for m in model.collection_centres:
#     for r in model.reprocessing_centres:
#         if model.g[m, r].value != 0:
#             print("collection_centres:", m, "reprocessing_centres:", r)
#             print(model.g[m, r].value)
#
#
# print("\nVariable DE")
# for s in model.design_alternatives:
#     for c in model.components:
#         if model.de[s,c].value != 0:
#             print("design:", s, "component:", c)
#             print(model.de[s,c].value)
#
#
#
# print("\nVariable ERP")
# for r in model.reprocessing_centres:
#     for j in model.plants:
#         for c in model.components:
#             if model.erp[r,j,c].value != 0:
#                 print("reprocessing_centres:", r, "plants:", j, "components:", c)
#                 print(model.erp[r,j,c].value)
#
#
# print("\nVariable ERM")
# for r in model.reprocessing_centres:
#     for j in model.plants:
#         for c in model.components:
#             if model.erm[r, j, c].value != 0:
#                 print("reprocessing_centres:", r, "plants:", j, "components:", c)
#                 print(model.erm[r, j, c].value)
#
#
# print("\nVariable ER")
# for r in model.reprocessing_centres:
#     for j in model.plants:
#         for c in model.components:
#             if model.er[r,j,c].value != 0:
#                 print("reprocessing_centres:", r, "plants:", j, "components:", c)
#                 print(model.er[r,j,c].value)
#
#
# print("\nVariable C")
# for r in model.reprocessing_centres:
#     for k in model.retailers:
#         if model.c[r, k].value != 0:
#             print("reprocessing_centres:", r, "retailers:", k)
#             print(model.c[r, k].value)
#
#
# print("\nVariable OPM")
# for m in model.collection_centres:
#     if model.opm[m].value != 0:
#         print("collection_centres:", m)
#         print(model.opm[m].value)

#
#
# print("\nVariable OPR")
# for r in model.reprocessing_centres:
#     if model.opr[r].value != 0:
#         print("reprocessing_centres:", r)
#         print(model.opr[r].value)
#
#
#
# print("\nVariable OPS")
# for i in model.suppliers:
#     if model.ops[i].value != 0:
#         print("suppliers:", i)
#         print(model.ops[i].value)
#
#
#
# print("\nVariable OPP")
# for j in model.plants:
#     if model.opp[j].value != 0:
#         print("plants:", j)
#         print(model.opp[j].value)


#
# print("\nVariable AR")
# for a in model.architectures:
#     if model.ar[a].value != 0:
#         print("architectures:", a)
#         print(model.ar[a].value)
#
#
#
# print("\nVariable RIMP")
# for r in model.r_imperatives:
#     if model.rimp[r].value != 0:
#         print("r_imperatives:", r)
#         print(model.rimp[r].value)
#
