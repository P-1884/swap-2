

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
@click.option('--report', default=None, help='Save report about analysis to this specified path, if provied.')
def run(name, data, trajectory=None, report=None):
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

    swap()  # score_users, apply_subjects, score_subjects
    logger.info('Retiring')
    swap.retire(config.fpr, config.mdr)
    if report is not None:
        logger.info('Reporting to {0}'.format(report))
        swap.report(path=report)
    if trajectory is not None:
        logger.info('Plotting some trajectories to {0}'.format(trajectory))
        trajectory_plot(swap=swap, path=trajectory)
    logger.info('entering interactive after applying classifications but before saving')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()

@ui.cli.command()
@click.argument('name')
@click.argument('data')
@click.option('--unsupervised', is_flag=True, default=False, help='Run offline SWAP algorithm updating confusion matrices using all objects in the catalog. Default: False')
@click.option('--ignore_gold_status', is_flag=True, default=False, help='Run offline SWAP algorithm ignoring gold status of objects in catalog, so that confusion matrices updated using golds uses current probability estimate. Default: False')
@click.option('--report', default=None, help='Save report about analysis to this specified path, if provied.')
def offline(name, data, unsupervised=False, ignore_gold_status=False, report=None):
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
    swap.retire(config.fpr, config.mdr)
    if report is not None:
        logger.info('Reporting to {0}'.format(report))
        swap.report(path=report)
    logger.info('entering interactive after applying classifications but before saving')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()


@ui.cli.command()
@click.argument('name')
@click.option('--config', is_flag=True)
def new(name, config):
    if config:
        config = Config()
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
    code.interact(local={**globals(), **locals()})

@ui.cli.command()
@click.argument('old_name')
@click.argument('new_name')
def copy(old_name, new_name):
    from swap.data import path
    swap = SWAP.load(old_name)
    new_name_path = path(new_name + '.pkl')
    logger.info('Saving new data at {0} after code interactive'.format(new_name_path))
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
    swap.report(path=report_path, report_classifications=False)
    logger.info('exporting score')
    score_path = directory + '/{0}_scores.csv'.format(swap.name)
    swap.export(path=score_path)


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
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()
