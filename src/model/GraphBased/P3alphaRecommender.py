#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize

from course_lib.Base.BaseRecommender import BaseRecommender
from course_lib.Base.Recommender_utils import check_matrix, similarityMatrixTopK

from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import time, sys


class P3alphaRecommender(BaseRecommender):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alphaRecommender"

    def __init__(self, URM_train, user_W_sparse, item_W_sparse, verbose=True):
        super(P3alphaRecommender, self).__init__(URM_train, verbose=verbose)
        self.user_W_sparse = user_W_sparse
        self.item_W_sparse = item_W_sparse


    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                                        self.min_rating,
                                                                                                        self.topK,
                                                                                                        self.implicit,
                                                                                                        self.normalize_similarity)

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        self.Pui = sps.hstack([self.user_W_sparse, self.URM_train], format="csr")
        self.Piu = sps.hstack([self.URM_train.T, self.item_W_sparse], format="csr")
        self.P = sps.vstack([self.Pui, self.Piu], format="csr")

        # Pui is the row-normalized urm
        Pui = normalize(self.P, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        #X_bool = self.URM_train.transpose(copy=True)
        X_bool = self.P.copy()
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # Set rows, cols, values
        rows = rows[:numCells]
        cols = cols[:numCells]
        values = values[:numCells]

        self.W_sparse = sps.csr_matrix((values, (rows, cols)),
                                       shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        # Get only items cols
        """item_col_mask = cols >= self.n_users
        rows = rows[item_col_mask]
        cols = cols[item_col_mask] - self.n_users
        values = values[item_col_mask]"""

        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
        user_profile_array = self.Pui[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()[:, self.n_users:]

        return item_scores
