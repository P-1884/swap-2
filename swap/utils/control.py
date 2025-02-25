
import pickle
import os

from swap.utils.subject import Subjects, ScoreStats, Thresholds
from swap.utils.user import Users
from swap.utils.plots import thresholds_setting
import swap.data

import logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

class Config:

    def __init__(self, **kwargs):
        annotation = {
            'task': 'T0',
            'value_key': None,
            'value_separator': '.',
            'true': [1],
            'false': [0],
        }
        if 'annotation' in kwargs:
            annotation.update(kwargs['annotation'])
        self.annotation = annotation
        self.p_retire_lens = kwargs.get('p_retire_lens', .9)
        self.p_retire_dud = kwargs.get('p_retire_dud', 1e-3)
        self.p_real, self.p_bogus = thresholds_setting()
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

        self.classifications = []

        self.seen_classifications = {}

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
            if data.get('seen_classifications'):
                swp.seen_classifications = data['seen_classifications']
        else:
            swp = SWAP(name)
        return swp

    def __call__(self):
#        logger.info('score users')
        self.score_users()
#        logger.info('apply subjects')
        self.apply_subjects()
#        logger.info('score_subjects')
        self.score_subjects()

    def offline(self, unsupervised=False, ignore_gold_status=False):
        # like __call__, but now we incorporate the probabilities of the unknown samples. In order to avoid breaking pieces of user and subject, I do the math here, and then apply the info
        import numpy as np

        logger.info('OfflineSwap: ignore_gold_status={0}, unsupervised={1}'.format(ignore_gold_status, unsupervised))

        # need to turn the classifications into a more easily manipulatable form

#        logger.debug('OfflineSwap: Getting scores')
        uids = []
        sids = []
        classifications = []
        for classification in self.classifications:
            user_id, subject_id, cl = classification

            if user_id not in uids:
                uids.append(user_id)

            if subject_id not in sids:
                sids.append(subject_id)
            subject = self.subjects[subject_id]
            classifications.append([uids.index(user_id), sids.index(subject_id), cl, subject.gold])
        classifications = np.array(classifications)

#        logger.debug('OfflineSwap: confusions')
        confusions = []
        for uid in uids:
            confusions.append(self.users[uid].score)
        confusions = np.array(confusions)  # confusions[:,0] == PD, confusions[:,1] == PL

#        logger.debug('OfflineSwap: probabilities')
        probabilities = []
        unknown_indices = []
        for i, sid in enumerate(sids):
            subject = self.subjects[sid]
            probabilities.append(subject.score)
            if subject.gold != -1:
                unknown_indices.append(i)
        unknown_indices = np.array(unknown_indices)
        probabilities = np.array(probabilities)

        # EM parameters
        N_min = 40
        N_max = 1000
        if not ignore_gold_status and not unsupervised:
            N_min = 2  # this should converge right away
        epsilon_min = 1e-8
        gamma = 1

        epsilon_taus = 10
        N_try = 0
#        logger.info('Running Expectation Maximization')
        while (epsilon_taus > epsilon_min) * (N_try < N_max) + (N_try < N_min):
            # collect old probabilities for assessing convergence
            old_probabilities = probabilities.copy()

            # E step
            p_real = np.zeros_like(probabilities)
            p_bogus = np.zeros_like(probabilities)
            n_observations = np.zeros_like(probabilities)
            n_user_observations = np.zeros(len(confusions))
            # this for loop could probably be done in a smarter fashion. Or at least in numba instead
            for ci in classifications:
                uid, sid, cid, gold = ci

                # TODO: I forget which of these is the bestest way
                # p0i = probabilities[sid]
                # p0i = np.mean(probabilities)
                p0i = subject.p0  # this converges much quicker
                p_real[sid] += (confusions[uid, 1] ** cid *
                                (1 - confusions[uid, 1]) ** (1 - cid) *
                                p0i)
                p_bogus[sid] += ((1 - confusions[uid, 0]) ** cid *
                                 confusions[uid, 0] ** (1 - cid) *
                                 (1 - p0i))
                n_observations[sid] += 1.
                n_user_observations[uid] += 1.

            # any subjects with no observations get put back to their prior score
            probabilities = np.where(n_observations > 0, p_real / (p_real + p_bogus) / n_observations, old_probabilities)

            # assess convergence
            epsilon_taus = np.sum(np.abs(probabilities - old_probabilities) / len(probabilities))

            # M step
            confusions_numer = np.zeros_like(confusions)
            confusions_denom = np.zeros_like(confusions)
            for ci in classifications:
                uid, sid, cid, gold = ci

                if not unsupervised and gold == -1:
                    # ignore non golds in supervised mode
                    continue

                if ignore_gold_status:
                    pi = probabilities[sid]
                else:
                    if gold == 0:
                        pi = 0
                    elif gold == 1:
                        pi = 1
                    else:
                        pi = probabilities[sid]

                confusions_numer[uid, 0] += (1 - cid) * (1 - pi)
                confusions_numer[uid, 1] += cid * pi
                confusions_denom[uid, 0] += 1 - pi
                confusions_denom[uid, 1] += pi

            confusions = (gamma + confusions_numer) / (2 * gamma + confusions_denom)

            N_try += 1
            logger.debug('EM Step {0} out of max {1}. Convergence Score: {2:.2e}.'.format(N_try, N_max, epsilon_taus))

        logger.info('Finished EM at Step {0}. Convergence Score: {1:.2e}'.format(N_try, epsilon_taus))

#        logger.info('score users')
        # apply scores
        for user in self.users.iter():
            # hacky: set prior, seen, and correct values, and truncate history
            # this is because we cannot set the score itself (it is a method that is considered a property of the user class)
            try:
                uid = uids.index(user.id)
            except ValueError:
                continue
            p_bogus, p_real = confusions[uid]
            # take total seen and translate this into that
            n_real = probabilities.sum()
            n_bogus = len(probabilities) - n_real

            user.seen = [n_bogus - 2 * gamma, n_real - 2 * gamma, user.seen[2]]
            user.correct = [n_bogus * p_bogus - gamma, n_real * p_real - gamma]
            # # truncate history
            # user.prior = (user.correct, user.seen)
            # user.history = []

#        logger.info('apply subjects')
        self.apply_subjects()

#        logger.info('score subjects')
        for subject in self.subjects.iter():
            # emulate update_score
            try:
                sid = sids.index(subject.id)
            except ValueError:
                continue
            probability = probabilities[sid]
            print(sid, subject.id, probability)
            # modify prior == score
            subject.score = probability
            # # truncate history for the truncate step
            # subject.prior = probability
            # subject.history = []

    def classify(self, user, subject, cl, id_):
        if self.last_id is None or id_ > self.last_id:
            self.last_id = id_

        # check if classification pair is in seen_classifications
        try:
            self.seen_classifications[(user, subject)]
            # we have already had this pair happen. ignore
#            logger.debug('Already saw {0} classify {1}. Ignoring Classification {2},{3}'.format(user, subject, cl, id_))
            return 0
        except KeyError:
            # have not seen this, so we may add this classification
            self.seen_classifications[(user, subject)] = True

        user = self.users[user]
        subject = self.subjects[subject]

        user.classify(subject, cl)
        subject.classify(user, cl)

        self.classifications.append([user.id, subject.id, cl])
        return 1

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
        for ui, u in enumerate(self.users.iter()):
#            logger.debug('User {0} of {1}: {2} with {3} classifications'.format(ui, len(self.users), u.id, len(u.history)))
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

    def retire(self, p_retire_dud, p_retire_lens):
        t = Thresholds(self.subjects, p_retire_dud, p_retire_lens)
        self.thresholds = t
        bogus, real = t()  # these are the threshold scores: p < bogus -> object is retired as bogus, and p > real -> object is retired as real.

        for subject in self.subjects.iter():
            # subject.update_score((bogus, real))
            subject.retire((bogus, real))

    def save(self, name=None):
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
            'seen_classifications': self.seen_classifications,
        }

        if name is None:
            name = swap.data.path(self.name + '.pkl')
        with open(name, 'wb') as file:
            pickle.dump(data, file)

    def report(self, path=None, report_subjects=True, report_users=True, report_classifications=True):
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
            # Target P_retire_dud, P_retire_lens, bogus and real thresholds
 #           p_bogus = self.thresholds.thresholds[0]
 #           p_real = self.thresholds.thresholds[1]
            p_real, p_bogus = thresholds_setting()

            report += '\nTarget P_retire_dud: {0:.3f}, Target P_retire_lens: {1:.3f}, P(Retire Bogus): {2:.3f}, P(Retire Real): {3:.3f}\n'.format(self.thresholds.p_retire_dud, self.thresholds.p_retire_lens, p_bogus, p_real)

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
            # report += 'p_retire_dud: {0:.4f}, mdf: {1:.4f}'.format(1 - real[1] / total[1], 1 - bogus[0] / total[0])
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
        if path is None:
            path = swap.data.path(self.name + '_report.txt')
        logger.info('Saving report to {0}'.format(path))
        with open(path, 'w') as file:
            file.write(report)

    def export_subjects(self, path=None):
        import csv

        if path is None:
            path = swap.data.path(self.name + '_scores.csv')
        logger.info('Saving subject scores to {0}'.format(path))

        with open(path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["id","gold","score","retired","seen"])
            for subject in self.subjects.iter():
                retired = subject.retired
                if retired is None:
                    retired = -1
                row = [subject.id, subject.gold, subject.score, retired, subject.seen]
                writer.writerow(row)

    def export_users(self, path=None):
        import csv

        if path is None:
            path = swap.data.path(self.name + '_skills.csv')
        logger.info('Saving user skills to {0}'.format(path))

        with open(path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["id","name","bogus_seen","real_seen","other_seen","bogus_correct","real_correct","PD","PL"])
            for user in self.users.iter():
                row = [user.id, user.name, user.seen[0], user.seen[1], user.seen[2], user.correct[0], user.correct[1], user.score[0], user.score[1]]
                writer.writerow(row)


    @property
    def performance(self):
        if self._performance is None:
            self._performance = ScoreStats(self.subjects, self.thresholds)
            self._performance()
        return self._performance
