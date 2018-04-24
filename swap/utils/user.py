
from collections import OrderedDict

from swap.utils.collection import Collection


class User:

    def __init__(self, user, username, correct, seen):
        self.id = user
        self.name = username
        self.history = []

        self.prior = (correct, seen)
        self.correct = correct
        self.seen = seen

    @classmethod
    def new(cls, user, username):
        return cls(user, username, [0, 0], [0, 0])

    def save(self):
        # save user to file
        pass

    def classify(self, subject, cl):
        # Add classification to history
        self.history.append((subject.id, subject.gold, cl))

    def update_subject(self, subject):
        for i in range(len(self.history)):
            h = self.history[i]
            if h[0] == subject.id:
                self.history[i] = (h[0], subject.gold, h[2])

    @property
    def score(self):
        correct = self.correct
        seen = self.seen
        gamma = 1  # TODO: this should be configurable

        score = [.5, .5]  # TODO: this should be configurable
        for i in [0, 1]:
            if seen[i] > 0:
                score[i] = (correct[i]+gamma) / (seen[i]+2*gamma)

        return score

    def update_score(self):
        correct = self.prior[0]
        seen = self.prior[1]
        for _, gold, cl in self.history:
            if gold in [0, 1]:
                seen[gold] += 1
                if gold == cl:
                    correct[gold] += 1

        self.seen = seen
        self.correct = correct
        return self.score

    def dump(self):
        return OrderedDict([
            ('user', self.id),
            ('username', self.name),
            ('correct', self.correct),
            ('seen', self.seen),
        ])

    def report(self, report_classifications=True):
        string = '# user id: {0}, name: {1},'.format(self.id, self.name)
        string += ' PBogus: {0:.3f}, PReal: {1:.3f}, length: {2}\n'.format(self.score[0], self.score[1], sum(self.seen))
        string += '## Bogus Seen: {0}, Bogus Correct: {1}, Real Seen: {2}, Real Correct: {3}\n'.format(self.seen[0], self.correct[0], self.seen[1], self.correct[1])
        if report_classifications:
            string += '# Subject ID, Gold, Classification\n'
            for s in self.history:
                id = s[0]
                gold = s[1]
                classification = s[2]
                string += '{0}, {1}, {2}\n'.format(id, gold, classification)
            string += '\n'

        return string

    def truncate(self):
        self.prior = (self.correct, self.seen)
        self.history = []

    @classmethod
    def load(cls, data):
        return cls(**data)

    def __str__(self):
        return 'id %s name %14s score %s length %d' % \
                (str(self.id), self.name, self.score, sum(self.seen))

    def __repr__(self):
        return str(self)


class Users(Collection):

    @staticmethod
    def new(user):
        return User.new(user, None)

    @classmethod
    def _load_item(cls, data):
        return User.load(data)
