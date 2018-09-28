

import click
import csv
import code
import sys

from swap.ui import ui
from swap.utils.control import SWAP, Config, Thresholds
from swap.utils.parser import ClassificationParser
from swap.utils.plots import trajectory_plot


import logging
logger = logging.getLogger(__name__)


@ui.cli.command()
@click.argument('name')
def clear(name):
    swap = SWAP.load(name)
    swap = SWAP(name, swap.config)
    swap.save()


@ui.cli.command()
@click.argument('name')
@click.argument('data')
@click.option('--trajectory', default=None, help='Save trajectories of a random subsample of subjects to this specified path, if provided.')
@click.option('--report', default=None, help='Save report about analysis to this specified path, if provided.')
@click.option('--scores', default=None, help='Save subject scores about analysis to this specified path, if provided.')
@click.option('--skills', default=None, help='Save user skills about analysis to this specified path, if provided.')
def run(name, data, trajectory=None, report=None, scores=None, skills=None):
    swap = SWAP.load(name)
    config = swap.config
    parser = ClassificationParser(config)

    classifications = 0
    classifications_ingested = 0
    with open(data, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            row = parser.parse(row)
            if row is None:
                logger.error(row)
                continue

            if i % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write("%d records processed\r" % i)

            did_classify = swap.classify(**row)  # appends to agent and subject histories
            classifications += 1
            classifications_ingested += did_classify

            # if i > 0 and i % 1e6 == 0:
                # print()
                # print('Applying records')
                # swap()
                # swap.truncate()
    logger.info('Processed {0} classifications, of which {1} were ingested as unique (user,subject) pairs'.format(classifications, classifications_ingested))

    swap()  # score_users, apply_subjects, score_subjects
    logger.info('Retiring')
    swap.retire(config.p_retire_dud, config.p_retire_lens)
    if report is not None:
        logger.info('Reporting to {0}'.format(report))
        swap.report(path=report)
    if trajectory is not None:
        logger.info('Plotting some trajectories to {0}'.format(trajectory))
        trajectory_plot(swap=swap, path=trajectory)
    if scores is not None:
        logger.info('exporting subject scores to {0}'.format(scores))
        swap.export_subjects(path=scores)
    if skills is not None:
        logger.info('exporting user skill to {0}'.format(skills))
        swap.export_users(path=skills)
    logger.info('entering interactive after applying classifications but before saving')
    logger.warning('\n#####Press ctrl-d to continue the code. Type exit() to stop the code')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()

@ui.cli.command()
@click.argument('name')
@click.argument('data')
@click.option('--unsupervised', is_flag=True, default=False, help='Run offline SWAP algorithm updating confusion matrices using all objects in the catalog. Default: False')
@click.option('--ignore_gold_status', is_flag=True, default=False, help='Run offline SWAP algorithm ignoring gold status of objects in catalog, so that confusion matrices updated using golds uses current probability estimate. Default: False')
@click.option('--report', default=None, help='Save report about analysis to this specified path, if provied.')
@click.option('--scores', default=None, help='Save exported scores about analysis to this specified path, if provided.')
@click.option('--skills', default=None, help='Save user skills about analysis to this specified path, if provided.')
def offline(name, data, unsupervised=False, ignore_gold_status=False, report=None, scores=None, skills=None):
    swap = SWAP.load(name)
    config = swap.config
    parser = ClassificationParser(config)

    with open(data, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            row = parser.parse(row)
            if row is None:
                logger.error(row)
                continue

            if i % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write("%d records processed\r" % i)

            swap.classify(**row)  # appends to agent and subject histories

            # if i > 0 and i % 1e6 == 0:
                # print()
                # print('Applying records')
                # swap()
                # swap.truncate()

    logger.info('Entering expectation_maximization')
    swap.offline(unsupervised=unsupervised, ignore_gold_status=ignore_gold_status)
    logger.info('Retiring')
    swap.retire(config.p_retire_dud, config.p_retire_lens)
    if report is not None:
        logger.info('Reporting to {0}'.format(report))
        swap.report(path=report)
    if scores is not None:
        logger.info('exporting subject scores to {0}'.format(scores))
        swap.export_subjects(path=scores)
    if skills is not None:
        logger.info('exporting user skills to {0}'.format(skills))
        swap.export_users(path=skills)
    logger.info('entering interactive after applying classifications but before saving')
    logger.warning('\n#####Press ctrl-d to continue the code. Type exit() to stop the code')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()


@ui.cli.command()
@click.argument('name')
@click.option('--config', is_flag=True)
def new(name, config):
    if config:
        config = Config()
        logger.warning('\n#####Press ctrl-d to continue the code. Type exit() to stop the code')
        code.interact(local=locals())
        swap = SWAP(name, config)
    else:
        swap = SWAP(name)
    logger.info('saving')
    swap.save()


@ui.cli.command()
@click.argument('name')
def load(name):
    swap = SWAP.load(name)
    logger.warning('\n#####Press ctrl-d to continue the code. Type exit() to stop the code')
    code.interact(local={**globals(), **locals()})

@ui.cli.command()
@click.argument('old_name')
@click.argument('new_name')
def copy(old_name, new_name):
    from swap.data import path
    swap = SWAP.load(old_name)
    new_name_path = path(new_name + '.pkl')
    logger.info('Saving new data at {0} after code interactive'.format(new_name_path))
    logger.warning('\n#####Press ctrl-d to continue the code. Type exit() to stop the code')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save(name=new_name_path)

@ui.cli.command()
@click.argument('name')
@click.argument('directory')
def export(name, directory):
    logger.info('loading swap {0}'.format(name))
    swap = SWAP.load(name)
    report_path = directory + '/{0}_report.txt'.format(swap.name)
    logger.info('reporting')
    swap.report(path=report_path, report_classifications=True)
    logger.info('exporting score')
    score_path = directory + '/{0}_scores.csv'.format(swap.name)
    swap.export_subjects(path=score_path)
    logger.info('exporting score')
    score_path = directory + '/{0}_skills.csv'.format(swap.name)
    swap.export_users(path=score_path)
    trajectory_path = directory + '/{0}_trajectory.pdf'.format(swap.name)
    logger.info('Plotting some trajectories to {0}'.format(trajectory_path))
    trajectory_plot(swap=swap, path=trajectory_path)


@ui.cli.command()
@click.argument('name')
@click.argument('path')
def golds(name, path):
    golds = []
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            golds.append((int(row['subject']), int(row['gold'])))

            if i % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write("%d records processed\r" % i)

    swap = SWAP.load(name)
    swap.apply_golds(golds)
    logger.info('entering interactive after applying golds but before saving')
    logger.warning('\n#####Press ctrl-d to continue the code. Type exit() to stop the code')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()
