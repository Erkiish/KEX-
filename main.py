from Data.Data_Class import DataSetGenerator
from MPT.Markovitz_Method import MarkovitzMethod, MarkovitzTest, MarkovitzGradientAscent

dataclass = DataSetGenerator('oliver')

monthly_data = dataclass.get_monthly_data('2012-01-01', '2015-01-01')

markovitz_class = MarkovitzMethod(monthly_data.iloc[:, 10:15])

omega_test = markovitz_class.expected_return
omega_test[:] = 1/len(omega_test)


markovitz_optimizer = MarkovitzGradientAscent(markovitz_class)
markovitz_optimizer.stochastic_optimization()


#markovitz_test_res = markovitz_test_class.run_test()
#print(markovitz_test_res)
#print(markovitz_test_res.describe())
