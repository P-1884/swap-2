
import json
import logging
logger = logging.getLogger(__name__)


class ClassificationParser:

    def __init__(self, config):
        self.annotation = AnnotationParser(config)

    def parse(self, cl):
        user = cl['user_id']
        if user == '':
            user = cl['user_name']
        else:
            user = int(user)

        annotation = self.annotation.parse(cl['annotations'])
        if annotation is None:
            logger.error('Skipping classification %s', cl)
            return None
        return {
            'user': user,
            'subject': int(cl['subject_ids']),
            'cl': annotation,
            'id_': int(cl['classification_id']),
        }


class AnnotationParser:

    def __init__(self, config):
        self.parser = config.annotation

    def parse(self, annotations):
        if type(annotations) is str:
            annotations = json.loads(annotations)
        annotation = self._find_task(annotations)
        if annotation is None:
            return None

        value = self._parse_value(annotation['value'])
        if value is None:
            task = self.parser['task']
            value_key = self.parser['value_key']

            logger.error('Error parsing annotation %s %s',
                         str(task), str(value_key))
            return None

        return value

    @staticmethod
    def _navigate(obj, dotkey, split='.'):
        steps = dotkey.split(split)
        item = obj

        for key in steps:
            if type(item) is list:
                key = int(key)

            item = item[key]

        return item

    def _find_task(self, annotations):
        """
        Find the right task from the annotation field in a classification

        Needs to be dynamic because csv dump and caesar stream send
        classifications with different formats
        """
        task = self.parser['task']
        if type(annotations) is dict and task in annotations:
            return annotations[task][0]

        if type(annotations) is list:
            for annotation in annotations:
                if annotation['task'] == task:
                    return annotation

        logger.error('Can\' find task %s', annotations)
        return None
        # raise self.AnnotationError(task, '', annotations)

    def _parse_value(self, value):
        """
        Parses the value field of an annotation task
        """
        key = self.parser['value_key']
        sep = self.parser['value_separator']

        if key is not None:
            value = self._navigate(value, key, sep)

        # CPD + PJM 09.04.18: if ANY annotation is made in spacewarps, then the
        # user must think it a positive classification. If None are made, then
        # they think it is a negative classification. If we were going to use
        # the original formatting, then all user classifications (positive or
        # negative) would need to contain some sort of positive/negative tag,
        # regardless of whether an annotation is made.
        if len(value) > 0:
            return 1
        elif len(value) == 0:
            return 0
        else:
            if value in self.config.true:
                return 1
            if value in self.config.false:
                return 0
