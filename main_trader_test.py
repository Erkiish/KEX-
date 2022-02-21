from Strategies.Strategies import TESTRSIouStrategy
from Data_Pipelines.Sim_Data_Pipeline import test_pipeline_x, sim_data_getter_x


test = sim_data_getter_x(5, 1000, std=0.1, drift=0.1)
print(test['4']['rsi_14'].min())