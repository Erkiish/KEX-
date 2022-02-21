from Strategies.Strategies import TESTRSIouStrategy
from Data_Pipelines.Sim_Data_Pipeline import test_pipeline_x, sim_data_getter_x


test = test_pipeline_x(50, 100000, TESTRSIouStrategy())
print(test['2'].loc[:, 'buy_signal'].sum())