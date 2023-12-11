import math
import scipy.sparse.linalg
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from load_data import load_data, getA, nr_users, nr_movies

class BaseLinePredictor:
    def train(self, data):
        A = getA(data)
        M = len(data)

        self.r_bar = data.sum(0)[2] / M

        C = data[:,2] - self.r_bar

        B = scipy.sparse.linalg.lsqr(A, C)[0]
        self.bu = B[:nr_users]
        self.bm = B[nr_users:]

    def predict(self, data):
        r_hat = np.zeros(len(data))
        for i in range(len(data)):
            user = data[i][0]
            movie = data[i][1]
            r_hat[i] = clamp(self.r_bar + self.bu[user] + self.bm[movie], 1, 5)

        return r_hat
    
    
class NeighborhoodPredictor(BaseLinePredictor):
    def __init__(self, u_min, L):
        super().__init__()
        self._u_min = u_min
        self._L = L

    def train(self, data):
        super().train(data)

        self.r_tilde = np.zeros((len(self.bu), len(self.bm))) # gives shape (2000, 1500) i.e 2000 rows of users and 1500 cols of movies
        for i in range(len(data)):
            data_index = data[i] # ith rating i.e (u, m, r) of index i
            user = data_index[0]
            movie = data_index[1]
            rating = data_index[2]
            self.r_tilde[user][movie] = rating - clamp(self.r_bar + self.bu[user] + self.bm[movie], 1, 5)

    def predict(self, data):
        r_hat = np.zeros(len(data))
        delta_matrix = self.delta_matrix()
        print("matrix done")
        last_movie = -1
        correction_denominator = 0
        for i in range(len(data)):
            user = data[i][0]
            movie = data[i][1]

            to_be_summed = []
            for m in range(nr_movies):
                if m != movie:
                    d_value = delta_matrix[movie][m]
                    numerator = d_value * self.r_tilde[user][m]
                    denominator = abs(d_value)
                    to_be_summed.append((numerator, denominator))
            
            to_be_summed = sorted(to_be_summed, key=lambda x: x[1], reverse=True)

            correction_term = 0
            correction_numerator = 0
            if movie != last_movie: # The denominator only changes when evaluating a different movie from last iteration
                correction_denominator = 0
                for k in range(len(to_be_summed)):
                    temp_num = to_be_summed[k][0]
                    temp_den = to_be_summed[k][1]
                    if k > self._L or temp_den == 0:
                        break
                    correction_numerator += temp_num
                    correction_denominator += temp_den
            else:
                for k in range(len(to_be_summed)):
                    temp_num = to_be_summed[k][0]
                    if k > self._L== 0:
                        break
                    correction_numerator += temp_num

            if correction_denominator != 0:
                correction_term = correction_numerator / correction_denominator

            last_movie = movie
            r_hat[i] = clamp(self.r_bar + self.bu[user] + self.bm[movie] + correction_term, 1, 5)
        return r_hat
    
    def delta(self, i, j, common_users):
        u = np.nonzero(common_users)
        if u[0].size < self._u_min:
            return 0
        num = np.transpose(self.r_tilde[:,i]) * self.r_tilde[:,j]
        den = np.linalg.norm(self.r_tilde[:,i][u]) * np.linalg.norm(self.r_tilde[:,j][u])
        result = np.sum(num) / den
        return result
    
    def delta_matrix(self):
        delta = np.zeros((nr_movies, nr_movies))
        A = np.zeros((nr_users, nr_movies))
        np.copyto(A, self.r_tilde, where=self.r_tilde != 0)
        for i in range(nr_movies): #rows
            for j in range(i+1, nr_movies, 1): #cols
                if i == j:
                    delta[i][j] = 1
                    continue
                col1 = A[:,i]
                col2 = A[:,j]
                common_users = np.multiply(col1, col2) #users with a rating of moviei and moviej in common

                delta[i][j] = self.delta(i, j, common_users)
                delta[j][i] = delta[i][j] # the matrix is symmetric
        return delta


def cal_RMSE(r_hat, r):
    summed = 0
    for i in range(len(r_hat)):
        summed += (r[i] - r_hat[i])**2
    return math.sqrt(summed / len(r_hat))

def draw_histogram(r_hat, r, r_hat_to_compare=None, title=""):
    err = abs(np.round(r_hat) - r)
    plt.figure()
    if r_hat_to_compare is not None:
        compare_err = abs(np.round(r_hat_to_compare) - r)
        plt.hist(compare_err, edgecolor="blue", bins=np.arange(-0.5, 5))
    
    plt.hist(err, edgecolor="black", bins=np.arange(-0.5, 5))
    plt.title(title)

def clamp(n, min, max): 
    if n < min: 
        return min
    elif n > max: 
        return max
    else: 
        return n 

if __name__ == '__main__':
    #filename = "task---2"
    filename = "thela038"

    training_data = load_data(filename + ".training")
    test_data = load_data(filename + ".test")

    print("-- Baseline predictor --")
    baseline_predictor = BaseLinePredictor()

    baseline_predictor.train(training_data)

    r_hat_baseline_training = baseline_predictor.predict(training_data)
    r_hat_baseline_test = baseline_predictor.predict(test_data)

    rmse_baseline_training = cal_RMSE(r_hat_baseline_training, training_data[:,2])
    rmse_baseline_test = cal_RMSE(r_hat_baseline_test, test_data[:,2])
    print("Training RMSE: {0:.3f}".format(rmse_baseline_training))
    print("Test RMSE: {0:.3f}".format(rmse_baseline_test))

    draw_histogram(r_hat_baseline_training, training_data[:,2], title="BaseLine Training")
    draw_histogram(r_hat_baseline_test, test_data[:,2], title="BaseLine Test")
    plt.show()

    u_min, L = 50, 100
    print("\n--- Movie neighborhood predictor with u_min = {} and L = {} ---".format(u_min, L))
    neighborhood_predictor = NeighborhoodPredictor(u_min=u_min, L=L)
    neighborhood_predictor.train(training_data)

    r_hat_neighborhood_training = neighborhood_predictor.predict(training_data)
    r_hat_neighborhood_test = neighborhood_predictor.predict(test_data)

    rmse_neighborhood_training = cal_RMSE(r_hat_neighborhood_training, training_data[:,2])
    rmse_neighborhood_test = cal_RMSE(r_hat_neighborhood_test, test_data[:,2])
    print("Training improvement: {0:.3f}%".format((rmse_baseline_training - rmse_neighborhood_training) / rmse_baseline_training * 100))
    print("Test improvement: {0:.3f}%".format((rmse_baseline_test - rmse_neighborhood_test) / rmse_baseline_test * 100))

    draw_histogram(r_hat_neighborhood_training, training_data[:,2], r_hat_baseline_training, title="Comparison training")
    draw_histogram(r_hat_neighborhood_test, test_data[:,2], r_hat_baseline_test, title="Comparison test")
    plt.show()
