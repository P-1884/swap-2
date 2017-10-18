
from swap.utils.scores import ScoreIterator, Score, ScoreExport
from swap.utils.golds import GoldGetter


class History:

    def __init__(self, id_, gold, score_history):
        """
        Parameters
        ----------
        id_ : int
            Subject it
        gold : int
            Subject gold label 1, 0, or -1
        scores : list
            List of score history for subject [0.2, 0.1, ...]
        """
        self.id = id_
        self.gold = gold
        self.scores = score_history

    def retire(self, thresholds):

        bogus, real = thresholds
        def check(score):
            return score < bogus and score > real

        for i, score in enumerate(self.scores):
            if check(score):
                return (i, score)

    def _retire_back(self, thresholds):

        bogus, real = thresholds
        def check(score):
            return score > bogus and score < real

        for i in reversed(range(len(self.scores))):
            score = self.scores[i]
            if check(score):
                i += 1
                if i >= len(self.scores):
                    return self.last
                return (i, self.scores[i])

    def last(self):
        return ((len(self.scores) - 1), self.scores[-1])


class HistoryExport:

    def __init__(self, history, gold_getter=None):
        """
        Parameters
        ----------
        history : {History}
            Mapping of Subject History to subject id
        """
        self.history = history

        if gold_getter is None:
            gold_getter = GoldGetter()
            gold_getter.all()

        self.gold_getter = gold_getter

    def get(self, id_):
        return self.history[id_]

    def traces(self):

        def func(history):
            return (history.gold, history.scores)

        return ScoreIterator(self.history, func)

    def score_export(self, thresholds=None, all_golds=False):
        scores = {}
        if thresholds is None:
            retire = lambda h: h.last()
        else:
            retire = lambda h: h.retire()

        for history in self.history.values():
            id_ = history.id
            n, p = retire(history)

            score = Score(id_, history.gold, p, ncl=n)
            scores[id_] = score

        return ScoreExport(
            scores, gold_getter=self.gold_getter,
            thresholds=thresholds, new_golds=all_golds)


    def __iter__(self):

        def func(history):
            return (history.id, history.gold, history.scores)
        return ScoreIterator(self.history, func)
