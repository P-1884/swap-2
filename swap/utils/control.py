
import pickle
import os

from swap.utils.subject import Subjects, ScoreStats, Thresholds
from swap.utils.user import Users
import swap.data

import logging
logger = logging.getLogger(__name__)


class Config:

    def __init__(self, **kwargs):
        annotation = {
            'task': 'T1',
            'value_key': None,
            'value_separator': '.',
            'true': [1],
            'false': [0],
        }
        if 'annotation' in kwargs:
            annotation.update(kwargs['annotation'])
        self.annotation = annotation
        self.mdr = kwargs.get('mdr', .1)
        self.fpr = kwargs.get('fpr', .01)

        self.online_name = kwargs.get('online_name', None)

    def dump(self):
        return self.__dict__.copy()

    @classmethod
    def load(cls, dump):
        config = cls()
        config.__dict__.update(dump)
        return config

    def __str__(self):
        return str(self.dump())

    def __repr__(self):
        return str(self)


class SWAP:

    def __init__(self, name, config=None):
        self.name = name
        self.users = Users()
        self.subjects = Subjects()

        if config is None:
            config = Config()
        self.config = config

        self.thresholds = None
        self._performance = None
        self.last_id = None

    @classmethod
    def load(cls, name):
        fname = name + '.pkl'

        path = swap.data.path(fname)
        if os.path.isfile(path):
            with open(swap.data.path(path), 'rb') as file:
                data = pickle.load(file)

            config = Config.load(data['config'])

            swp = SWAP(name, config)
            swp.last_id = data['last_id']
            swp.users = Users.load(data['users'])
            swp.subjects = Subjects.load(data['subjects'])

            if data.get('thresholds'):
                swp.thresholds = Thresholds.load(
                    swp.subjects, data['thresholds'])
        else:
            swp = SWAP(name)
        return swp

    def __call__(self):
        logger.info('score users')
        self.score_users()
        logger.info('apply subjects')
        self.apply_subjects()
        logger.info('score_subjects')
        self.score_subjects()

    def classify(self, user, subject, cl, id_):
        if self.last_id is None or id_ > self.last_id:
            self.last_id = id_

        user = self.users[user]
        subject = self.subjects[subject]

        user.classify(subject, cl)
        subject.classify(user, cl)

    def truncate(self):
        self.users.truncate()
        self.subjects.truncate()

    def score_users(self):
        for u in self.users.iter():
            u.update_score()

    def score_subjects(self):
        for s in self.subjects.iter():
            s.update_score()

    def apply_subjects(self):
        # update user scores to each subject
        for u in self.users.iter():
            for subject, _, _ in u.history:
                self.subjects[subject].update_user(u)

    def apply_gold(self, subject, gold):
        # update gold subjects to each user
        subject = self.subjects[subject]
        subject.gold = gold
        for user, _, _ in subject.history:
            self.users[user].update_subject(subject)

    def apply_golds(self, golds):
        for subject, gold in golds:
            self.apply_gold(subject, gold)

    def retire(self, fpr, mdr):
        t = Thresholds(self.subjects, fpr, mdr)
        self.thresholds = t
        bogus, real = t()  # these are the threshold scores: p < bogus -> object is retired as bogus, and p > real -> object is retired as real.

        for subject in self.subjects.iter():
            subject.update_score((bogus, real))

    def save(self):
        if self.thresholds is not None:
            thresholds = self.thresholds.dump()
        else:
            thresholds = None
        data = {
            'config': self.config.dump(),
            'users': self.users.dump(),
            'subjects': self.subjects.dump(),
            'thresholds': thresholds,
            'last_id': self.last_id,
        }

        fname = self.name + '.pkl'
        with open(swap.data.path(fname), 'wb') as file:
            pickle.dump(data, file)

    def report(self, report_subjects=True, report_users=True, report_classifications=True):
        # make a string for reporting, dumps to a text file located in same directory as pickle
        report = 'Report for SWAP Database {0}\n'.format(self.name)

        # number of subjects, users, and number of classifications
        # looks like the closest proxy we can easily and reliably have to the number of classifications is to count the number of seen in the subjects
        n_seen = 0
        for key in self.subjects.keys():
            subject = self.subjects[key]
            n_seen += subject.seen
        report += '\n{0} Subjects, {1} Users, {2} Classifications\n'.format(len(self.subjects), len(self.users), n_seen)

        if self.thresholds is not None:
            # Target FPR, MDR, bogus and real thresholds
            p_bogus = self.thresholds.thresholds[0]
            p_real = self.thresholds.thresholds[1]
            report += '\nTarget FPR: {0:.3f}, Target MDR: {1:.3f}, P(Retire Bogus): {2:.3f}, P(Retire Real): {3:.3f}\n'.format(self.thresholds.fpr, self.thresholds.mdr, p_bogus, p_real)

            # TODO: golds: give breakdown of golds classifications
            scores = self.thresholds.get_scores()
            total = [0, 0, 0]
            bogus = [0, 0, 0]
            real = [0, 0, 0]
            inconclusive = [0, 0, 0]
            for score in scores:
                index = {0: 0, 1: 1, -1: 2}[score[0]]
                total[index] += 1
                p = score[1]
                if p >= p_real:
                    real[index] += 1
                elif p <= p_bogus:
                    bogus[index] += 1
                else:
                    inconclusive[index] += 1
            for k, label in enumerate(['bogus', 'real', 'unknown']):
                report += '{0} {1}: {2} classified real, {3} classified bogus, {4} inconclusive'.format(label, total[k], real[k], bogus[k], inconclusive[k])
                report += '\n'
        else:
            logger.debug('skipping threshold reporting')


        if report_subjects:
            report += '\n#####\n# Subjects\n#####\n'
            for key in self.subjects.keys():
                subject = self.subjects[key]
                subject_report = subject.report(report_classifications=report_classifications)
                report += subject_report

        if report_users:
            report += '\n#####\n# Users\n#####\n\n'
            for key in self.users.keys():
                user = self.users[key]
                user_report = user.report(report_classifications=report_classifications)
                report += user_report

        # save it
        fname = self.name + '_report.txt'
        logger.info('Saving report to {0}'.format(swap.data.path(fname)))
        with open(swap.data.path(fname), 'w') as file:
            file.write(report)

    @property
    def performance(self):
        if self._performance is None:
            self._performance = ScoreStats(self.subjects, self.thresholds)
            self._performance()
        return self._performance
