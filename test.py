import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from recommender_model import RecSysSVD
import recsys_helper as rh
from component import sqlite

import unittest

class RecSysTest(unittest.TestCase):
    
    def setUp(self):
    
        self.data = pd.read_csv('data\movie_data.csv')
        self.data = self.data.rename({'userId':'user_id', 'movieId':'item_id', 'rating':'score'}, axis='columns')
        self.X, self.y = rh.create_train_data(self.data, 150)
        self.mdl = RecSysSVD()
        self.mdl.fit(self.X,self.y)
        
        
        
    def test_untitled_user_item(self):
        
        true_value = 0.3
        user = 611
        item = 1
        
        test_value = self.mdl.predict_one(user, item)
        self.assertEqual(test_value, true_value)
        
        
        
    def test_exist_user_item_min(self):
        
        true_value = 0
        user = 610
        item = 1
        
        test_value = self.mdl.predict_one(user, item)
        self.assertTrue(test_value > true_value)    
        
        
        
    def test_exist_user_item_max(self):
        
        true_value = 5
        user = 610
        item = 1
        
        test_value = self.mdl.predict_one(user, item)
        self.assertTrue(test_value <= true_value)  
        
        
        
    def test_fit(self):
        
        true_value = tuple([len(set(self.X.user_id)),len(set(self.X.item_id))])
        test_value = self.mdl.scores_table.shape  
    
        self.assertEqual(test_value, true_value)
        
        
        
    def test_X_table_type(self):
        
        true_value = 'int'
        test_value = str(self.X.user_id.dtype)[:3] == 'int'
        self.assertTrue(test_value, true_value)
        
        
        
        
        
if __name__ == '__main__':
    unittest.main()