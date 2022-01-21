import pandas as pd
import random
from typing import Union
class MarkovitzMethod:

    def __init__(self, data: pd.DataFrame):

        self.data = data
        self.data_pct = self.data.pct_change().iloc[1:, :]
        self.tickers = list(self.data.columns)
        self._calculate_expected_return()
        self._calculate_variance()
        self._calculate_covariance()

    def _calculate_expected_return(self):
        
        self.expected_return = self.data_pct.mean()

    def _calculate_variance(self):
        
        self.variance = self.data_pct.var()

    def _calculate_covariance(self):
        
        self.covariance = self.data_pct.cov()
    
    def calculate_portfolio_expected_return(self, portfolio_weights: pd.Series) -> float:

        return portfolio_weights.T.dot(self.expected_return)
    
    def calculate_portfolio_variance(self, portfolio_weights: pd.Series) -> float:

        res = self.covariance.dot(portfolio_weights)

        return portfolio_weights.T.dot(res)
    
    def calculate_optimization_function_gradient(self, portfolio_weights: pd.Series):

        self.gradient = portfolio_weights

        for ticker in self.tickers:

            self.gradient[ticker] = self.expected_return[ticker] - (2*portfolio_weights[ticker]*self.covariance.loc[ticker, ticker] + sum(self.covariance.drop(ticker, axis=1).loc[ticker, :]*portfolio_weights.drop(ticker)) \
                + sum(portfolio_weights.drop(ticker)*self.covariance.drop(ticker, axis=0).loc[:, ticker]))
        
        return self.gradient

class MarkovitzTest:

    def __init__(self, markovitz_class: MarkovitzMethod):

        self.markovitz_class = markovitz_class
        self.omega = markovitz_class.expected_return
        self.omega_len = len(self.omega)
        self.tickers = list(self.omega.index)
    
    def run_test(self):
        
        return self._iterate_omega_i()

    def _iterate_omega_i(self) -> pd.DataFrame:

        omega_result_dict = {
            ticker: self._iterate_omega(ticker) for ticker in self.tickers
        }

        return pd.DataFrame.from_dict(omega_result_dict, orient='index')

    def _iterate_omega(self, ticker: str) -> float:

        omega_x_data = {}
        
        for i in range(self.omega_len):
            omega_x = self.omega
            omega_x.loc[:] = i/self.omega_len*1/(self.omega_len - 1)
            omega_x[ticker] = 1 - i/self.omega_len

            portfolio_expected_return = self.markovitz_class.calculate_portfolio_expected_return(omega_x)
            portfolio_variance = self.markovitz_class.calculate_portfolio_variance(omega_x)

            omega_x_data[str(omega_x[ticker])] = portfolio_expected_return - portfolio_variance
        
        return omega_x_data


class MarkovitzGradientAscent:

    def __init__(self, markovitz_class: MarkovitzMethod):

        self.markovitz_class = markovitz_class
    
    def optimize(self, n_iter: int=100, start_omega: Union[bool, pd.Series]=False) -> tuple[pd.Series, float]:

        if isinstance(start_omega, bool):
            omega_x = self.markovitz_class.expected_return
            omega_x[:] = 1/len(omega_x)
        else:
            omega_x = start_omega
        
        portfolio_expected_return = self.markovitz_class.calculate_portfolio_expected_return(omega_x)
        portfolio_variance = self.markovitz_class.calculate_portfolio_variance(omega_x)
        optimization_result = portfolio_expected_return - portfolio_variance
        print("STARTING VALUE:")
        print(optimization_result)

        for _ in range(n_iter):

            gradient = self.markovitz_class.calculate_optimization_function_gradient(omega_x)

            omega_x = (omega_x + gradient)/(sum(omega_x.abs() + gradient.abs()))

        portfolio_expected_return = self.markovitz_class.calculate_portfolio_expected_return(omega_x)
        portfolio_variance = self.markovitz_class.calculate_portfolio_variance(omega_x)
        optimization_result = portfolio_expected_return - portfolio_variance
        print("OPTIMIZED VALUE:")
        print(optimization_result)

        return omega_x, optimization_result

    
    def stochastic_optimization(self, n_iter: int=10):

        local_maxima_omega_dict = {}
        top_maxima_index = 0
        top_maxima = 0

        tickers = list(self.markovitz_class.expected_return.index)
        num_tickers = len(tickers)

        for index, ticker in enumerate(tickers):

            start_omega = self.markovitz_class.expected_return
            start_omega.loc[:] = (1/100)/(num_tickers - 1)
            start_omega[ticker] = 1 - 1/100
            omega_res, res = self.optimize(n_iter=30, start_omega=start_omega)

            local_maxima_omega_dict[index] = {'omega': omega_res, 'res':res}

            if res > top_maxima:
                top_maxima = res
                top_maxima_index = index
        
        print(top_maxima)
        print(local_maxima_omega_dict[top_maxima_index]['omega'])



        


