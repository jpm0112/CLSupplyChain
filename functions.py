def generate_instance():


# parameters
np.random.seed(1048596)
# Initialize capacities with random values within a sensible range
suppliers_capacity = np.random.randint(400, 500, (supplier_number, parts_number))
np.savetxt('suppliers_capacity.csv', suppliers_capacity, delimiter=",")
plants_capacity = np.random.randint(500, 800, (plants_number))
np.savetxt('plants_capacity.csv', plants_capacity, delimiter=",")
retailer_demand = np.random.randint(200, 300, (retailers_number, periods_number))
np.savetxt('retailer_demand.csv', retailer_demand, delimiter=",")
collection_centres_capacity = np.random.randint(300, 500, (collection_centres_number, parts_number))
np.savetxt('collection_centres_capacity.csv', collection_centres_capacity, delimiter=",")
disassembly_centres_capacity = np.random.randint(300, 400, (disassembly_centres_number, parts_number))
np.savetxt('disassembly_centres_capacity.csv', disassembly_centres_capacity, delimiter=",")
remanufacturing_centres_capacity = np.random.randint(300, 400, (remanufacturing_centres_number, parts_number))
np.savetxt('remanufacturing_centres_capacity.csv', remanufacturing_centres_capacity, delimiter=",")

flow_cost_suppliers_plants = np.random.randint(1, 100,
                                               (supplier_number, plants_number))  # suppliers as rows, plants as columns
np.savetxt('flow_cost_suppliers_plants.csv', flow_cost_suppliers_plants, delimiter=",")
purchase_cost_suppliers = np.random.randint(1, 100,
                                            (parts_number, designs_number))  # suppliers as rows, plants as columns
    np.savetxt('purchase_cost_suppliers.csv', purchase_cost_suppliers, delimiter=",")
    flow_cost_plants_retailers = np.random.randint(1, 100, (plants_number, retailers_number))
    np.savetxt('flow_cost_plants_retailers.csv', flow_cost_plants_retailers, delimiter=",")
    flow_cost_retailers_collection_centres = np.random.randint(1, 50, (retailers_number, collection_centres_number))
    np.savetxt('flow_cost_retailers_collection_centres.csv', flow_cost_retailers_collection_centres, delimiter=",")
    flow_cost_collection_retailer = np.random.randint(1, 50, (collection_centres_number, retailers_number))
    np.savetxt('flow_cost_collection_retailer.csv', flow_cost_collection_retailer, delimiter=",")

    flow_cost_collection_centres_remanufacturing = np.random.randint(1, 100, (
    collection_centres_number, remanufacturing_centres_number))
    np.savetxt('flow_cost_collection_centres_remanufacturing.csv', flow_cost_collection_centres_remanufacturing,
           delimiter=",")

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
    parts_of_architecture = np.random.randint(1, 2, (architecture_number,
                                                 parts_number))  # rows are the architectures and columns are the part, replaces the r parameter in the model
    np.savetxt('parts_of_architecture.csv', parts_of_architecture, delimiter=",")
    r_imperatives_of_architecture = np.random.randint(0, 2, (architecture_number,
                                                         r_imperatives_number))  # rows are architectures and columns the part. It has a value of 1 if the r-imperative is possible with the architecutre
    np.savetxt('r_imperatives_of_architecture.csv', r_imperatives_of_architecture, delimiter=",")
    r_imperatives_of_designs = np.random.randint(0, 2, (designs_number,
                                                    r_imperatives_number))  # rows are designs and columns the part. It has a value of 1 if the r-imperative is possible with the design
    np.savetxt('r_imperatives_of_designs.csv', r_imperatives_of_designs, delimiter=",")
    designs_of_architecture = np.random.randint(0, 2, (architecture_number,
                                                   designs_number))  # rows are architecture and columns the design. It has a value of 1 if the design is possible with the architecture
    np.savetxt('designs_of_architecture.csv', designs_of_architecture, delimiter=",")
    designs_of_parts = np.random.randint(1, 2, (parts_number,
                                            designs_number))  # rows are parts and columns the design. It has a value of 1 if the design is possible with the part
    np.savetxt('designs_of_parts.csv', designs_of_parts, delimiter=",")
    return()


