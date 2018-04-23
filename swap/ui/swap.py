

import click
import csv
import code
import sys

from swap.ui import ui
from swap.utils.control import SWAP, Config, Thresholds
from swap.utils.parser import ClassificationParser

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
def run(name, data):
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
    logger.info('Reporting')
    swap.report()
    logger.info('entering interactive after applying classifications but before saving')
    code.interact(local={**globals(), **locals()})
    logger.info('saving')
    swap.save()

@ui.cli.command()
@click.argument('name')
@click.argument('data')
@click.option('--unsupervised', is_flag=True, default=False, help='Run offline SWAP algorithm updating confusion matrices using all objects in the catalog. Default: False')
@click.option('--ignore_gold_status', is_flag=True, default=False, help='Run offline SWAP algorithm ignoring gold status of objects in catalog, so that confusion matrices updated using golds uses current probability estimate. Default: False')
def offline(name, data, unsupervised=False, ignore_gold_status=False):
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
    logger.info('Reporting')
    swap.report()
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
