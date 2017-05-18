#!/usr/bin/env python
################################################################
# Script to test control functionality

import swap.db.classifications as dbcl
from swap.control import Control
from swap.control import GoldGetter

from unittest.mock import MagicMock, patch
import pytest

fields = {'user_id', 'classification_id', 'subject_id',
          'annotation', 'gold_label'}


class TestControl:

    @patch.object(dbcl, 'getRandomGoldSample', MagicMock(return_value=[]))
    def test_with_train_split(self):

        c = Control()
        c.gold_getter.random(100)
        c.get_gold_labels()

        dbcl.getRandomGoldSample.assert_called_with(100)

    @patch.object(dbcl, 'getAllGolds', MagicMock(return_value=[]))
    def test_without_train_split(self):
        c = Control()
        c.get_gold_labels()

        dbcl.getAllGolds.assert_called_with()


# def test_classifications_projection():
#     q = Query()
#     q.fields(['user_id', 'classification_id'])
#     raw = dbcl.classifications.aggregate(q.build())

#     item = raw.next()

#     assert 'user_id' in item
#     assert 'classification_id' in item
#     assert 'subject_id' not in item


# def test_classifications_limit():
#     q = Query()
#     q.fields(fields).limit(5)
#     raw = dbcl.classifications.aggregate(q.build())

#     assert len(list(raw)) == 5


# def test_users():
#     control = swap.Server(.5, .5)
#     users = control.getClassificationsByUser()

#     pprint(list(users))


@pytest.mark.skip(reason='Network call, takes too long')
def test_get_one_classification():
    """ Get the first classification
    """
    control = Control(0.5, 0.5)

    cursor = control.getClassifications()
    n_class = len(cursor)
    cl = cursor.next()

    assert n_class > 0
    assert type(cl) == dict
    assert len(cl) > 0


class TestGoldGetter:

    def test_wrapper_golds_to_None(self):
        old = dbcl.getAllGolds
        dbcl.getAllGolds = MagicMock(return_value=[])

        gg = GoldGetter()
        gg._golds = {}
        gg.all()

        assert gg._golds is None

        dbcl.getAllGolds = old

    def test_wrapper_getter(self):
        old = dbcl.getAllGolds
        dbcl.getAllGolds = MagicMock(return_value=[])

        gg = GoldGetter()
        gg._golds = {}
        gg.all()

        print(gg.getters)
        assert callable(gg.getters[0])

        dbcl.getAllGolds = old

    def test_getter_propagation(self):
        c = Control()
        c.gold_getter.getters = [lambda: {1: 1, 2: 0}]

        c.init_swap()
        assert c.swap.subjects.get(1).gold == 1
        assert c.swap.subjects.get(2).gold == 0

    def test_multiple_getters(self):
        c = Control()
        c.gold_getter.getters = [
            lambda: {1: 1, 2: 0},
            lambda: {3: 0, 4: 0}
        ]

        c.init_swap()
        assert c.swap.subjects.get(1).gold == 1
        assert c.swap.subjects.get(2).gold == 0
        assert c.swap.subjects.get(3).gold == 0
        assert c.swap.subjects.get(4).gold == 0

    @pytest.mark.skip(reason='Network call, takes too long')
    def test_real_multiple_getters(self):
        c = Control()
        gg = c.gold_getter

        gg.controversial(10)
        gg.consensus(10)

        golds = c.getGoldLabels()
        print(golds)
        assert len(golds) == 20
