from collections import OrderedDict
from itertools import chain
from django.db import models
from django.db.models.query import QuerySet, BaseIterable, RelatedPopulator
import django.db.models.fields.related

from django.db.models import sql, DEFERRED, Manager
from django.db.models.manager import BaseManager
from django.db.models.query_utils import FilteredRelation, Q

from abc import abstractmethod

from django.core import exceptions
from django.core.exceptions import FieldError
from django.db.models.expressions import Func, Star
from django.db.models.fields import DecimalField, FloatField, IntegerField
from django.db.models.constants import LOOKUP_SEP

from django.utils.functional import cached_property, partition

from functools import lru_cache

__all__ = [
    'Aggregate', 'Avg', 'Count', 'Max', 'Min', 'StdDev', 'Sum', 'Variance',
]

# The maximum number of items to display in a QuerySet.__repr__
REPR_OUTPUT_SIZE = 20

# Pull into this namespace for backwards compatibility.
EmptyResultSet = sql.EmptyResultSet

policy_applied = []

class ModelIterable(BaseIterable):
    """Iterable that yields a model instance for each row."""

    def __iter__(self):
        queryset = self.queryset
        if queryset.query.can_filter():
            queryset = queryset.model.apply_policy(queryset)
            self.queryset = queryset

        db = queryset.db
        compiler = queryset.query.get_compiler(using=db)
        # Execute the query. This will also fill compiler.select, klass_info,
        # and annotations.
        results = compiler.execute_sql(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)
        select, klass_info, annotation_col_map = (compiler.select, compiler.klass_info,
                                                  compiler.annotation_col_map)
        model_cls = klass_info['model']
        select_fields = klass_info['select_fields']
        model_fields_start, model_fields_end = select_fields[0], select_fields[-1] + 1
        init_list = [f[0].target.attname
                     for f in select[model_fields_start:model_fields_end]]
        related_populators = get_related_populators(klass_info, select, db)
        for row in compiler.results_iter(results):
            obj = model_cls.from_db(db, init_list, row[model_fields_start:model_fields_end])
            for rel_populator in related_populators:
                rel_populator.populate(row, obj)
            if annotation_col_map:
                for attr_name, col_pos in annotation_col_map.items():
                    setattr(obj, attr_name, row[col_pos])

            # Add the known related objects to the model, if there are any
            if queryset._known_related_objects:
                for field, rel_objs in queryset._known_related_objects.items():
                    # Avoid overwriting objects loaded e.g. by select_related
                    if field.is_cached(obj):
                        continue
                    pk = getattr(obj, field.get_attname())
                    try:
                        rel_obj = rel_objs[pk]
                    except KeyError:
                        pass  # may happen in qs1 | qs2 scenarios
                    else:
                        setattr(obj, field.name, rel_obj)

            yield obj


class ValuesIterable(BaseIterable):
    """
    Iterable returned by QuerySet.values() that yields a dict for each row.
    """

    def __iter__(self):
        queryset = self.queryset
        if queryset.query.can_filter():
            queryset = queryset.model.apply_policy(queryset)
            self.queryset = queryset

        query = queryset.query
        compiler = query.get_compiler(queryset.db)

        field_names = list(query.values_select)
        extra_names = list(query.extra_select)
        annotation_names = list(query.annotation_select)

        # extra(select=...) cols are always at the start of the row.
        names = extra_names + field_names + annotation_names

        indexes = range(len(names))
        for row in compiler.results_iter(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size):
            yield {names[i]: row[i] for i in indexes}


class ValuesListIterable(BaseIterable):
    """
    Iterable returned by QuerySet.values_list(flat=False) that yields a tuple
    for each row.
    """

    def __iter__(self):
        queryset = self.queryset
        if queryset.query.can_filter():
            queryset = queryset.model.apply_policy(queryset)
            self.queryset = queryset
            
        query = queryset.query
        compiler = query.get_compiler(queryset.db)

        if queryset._fields:
            field_names = list(query.values_select)
            extra_names = list(query.extra_select)
            annotation_names = list(query.annotation_select)

            # extra(select=...) cols are always at the start of the row.
            names = extra_names + field_names + annotation_names

            fields = list(queryset._fields) + [f for f in annotation_names if f not in queryset._fields]
            if fields != names:
                # Reorder according to fields.
                index_map = {name: idx for idx, name in enumerate(names)}
                rowfactory = operator.itemgetter(*[index_map[f] for f in fields])
                return map(
                    rowfactory,
                    compiler.results_iter(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)
                )
        return compiler.results_iter(tuple_expected=True, chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)


class NamedValuesListIterable(ValuesListIterable):
    """
    Iterable returned by QuerySet.values_list(named=True) that yields a
    namedtuple for each row.
    """

    @staticmethod
    @lru_cache()
    def create_namedtuple_class(*names):
        # Cache namedtuple() with @lru_cache() since it's too slow to be
        # called for every QuerySet evaluation.
        return namedtuple('Row', names)

    def __iter__(self):
        queryset = self.queryset
        if queryset.query.can_filter():
            queryset = queryset.model.apply_policy(queryset)
            self.queryset = queryset
            
        if queryset._fields:
            names = queryset._fields
        else:
            query = queryset.query
            names = [*query.extra_select, *query.values_select, *query.annotation_select]
        tuple_class = self.create_namedtuple_class(*names)
        new = tuple.__new__
        for row in super().__iter__():
            yield new(tuple_class, row)


class FlatValuesListIterable(BaseIterable):
    """
    Iterable returned by QuerySet.values_list(flat=True) that yields single
    values.
    """

    def __iter__(self):
        queryset = self.queryset
        if queryset.query.can_filter():
            queryset = queryset.model.apply_policy(queryset)
            self.queryset = queryset
            
        compiler = queryset.query.get_compiler(queryset.db)
        for row in compiler.results_iter(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size):
            yield row[0]
            
class EstrelaQuerySet(QuerySet):
    """Represent a lazy database lookup for a set of objects."""

    def __init__(self, model=None, query=None, using=None, hints=None):
        self.model = model
        self._db = using
        self._hints = hints or {}
        self.query = query or sql.Query(self.model)
        self._result_cache = None
        self._sticky_filter = False
        self._for_write = False
        self._prefetch_related_lookups = ()
        self._prefetch_done = False
        self._known_related_objects = {}  # {rel_field: {pk: rel_obj}}
        self._iterable_class = ModelIterable
        self._fields = None

    def as_manager(cls):
        manager = EstrelaManager.from_queryset(cls)()
        manager._built_with_as_manager = True
        return manager
    as_manager.queryset_only = True
    as_manager = classmethod(as_manager)
        
    def __repr__(self):
        data = list(self[:REPR_OUTPUT_SIZE + 1])
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = "...(remaining elements truncated)..."
        return '<%s %r>' % (self.__class__.__name__, data)

    def __len__(self):
        self._fetch_all()
        res = len(self._result_cache)
        return res

    def __iter__(self):
        """
        The queryset iterator protocol uses three nested iterators in the
        default case:
            1. sql.compiler.execute_sql()
               - Returns 100 rows at time (constants.GET_ITERATOR_CHUNK_SIZE)
                 using cursor.fetchmany(). This part is responsible for
                 doing some column masking, and returning the rows in chunks.
            2. sql.compiler.results_iter()
               - Returns one row at time. At this point the rows are still just
                 tuples. In some cases the return values are converted to
                 Python values at this location.
            3. self.iterator()
               - Responsible for turning the rows into model objects.
        """
        self._fetch_all()
        res = self.model.apply_post_policy(self._result_cache, self._fields)
        return iter(res)

    def __bool__(self):
        self._fetch_all()
        res = bool(self._result_cache)
        return res

    def __getitem__(self, k):
        """Retrieve an item or slice from the set of results."""
        if not isinstance(k, (int, slice)):
            raise TypeError
        assert ((not isinstance(k, slice) and (k >= 0)) or
                (isinstance(k, slice) and (k.start is None or k.start >= 0) and
                 (k.stop is None or k.stop >= 0))), \
            "Negative indexing is not supported."
        
        if self._result_cache is not None:
            return self._result_cache[k]
        
        if self.query.can_filter():
            self = self.model.apply_policy(self)
            
        if isinstance(k, slice):
            qs = self._chain()
            if k.start is not None:
                start = int(k.start)
            else:
                start = None
            if k.stop is not None:
                stop = int(k.stop)
            else:
                stop = None
            qs.query.set_limits(start, stop)
            return list(qs)[::k.step] if k.step else qs

        qs = self._chain()
        qs.query.set_limits(k, k + 1)
        qs._fetch_all()
        res = qs._result_cache[0]
        return res

    def __and__(self, other):
        self._merge_sanity_check(other)
        if isinstance(other, EmptyQuerySet):
            return other
        if isinstance(self, EmptyQuerySet):
            return self
        combined = self._chain()
        combined._merge_known_related_objects(other)
        combined.query.combine(other.query, sql.AND)
        return combined

    def __or__(self, other):
        self._merge_sanity_check(other)
        if isinstance(self, EmptyQuerySet):
            return other
        if isinstance(other, EmptyQuerySet):
            return self
        combined = self._chain()
        combined._merge_known_related_objects(other)
        combined.query.combine(other.query, sql.OR)
        return combined

    ####################################
    # METHODS THAT DO DATABASE QUERIES #
    ####################################

    def aggregate(self, *args, **kwargs):
        """
        Return a dictionary containing the calculations (aggregation)
        over the current queryset.

        If args is present the expression is passed as a kwarg using
        the Aggregate object's default alias.
        """
        if self.query.distinct_fields:
            raise NotImplementedError("aggregate() + distinct(fields) not implemented.")
        self._validate_values_are_expressions(args + tuple(kwargs.values()), method_name='aggregate')
        for arg in args:
            # The default_alias property raises TypeError if default_alias
            # can't be set automatically or AttributeError if it isn't an
            # attribute.
            try:
                arg.default_alias
            except (AttributeError, TypeError):
                raise TypeError("Complex aggregates require an alias")
            kwargs[arg.default_alias] = arg
            
        if self.query.can_filter():
            self = self.model.apply_policy(self, **kwargs)
        query = self.query.chain()
        for (alias, aggregate_expr) in kwargs.items():
            query.add_annotation(aggregate_expr, alias, is_summary=True)
            if not query.annotations[alias].contains_aggregate:
                raise TypeError("%s is not an aggregate expression" % alias)
        q = query.get_aggregation(self.db, kwargs)
        print(q)
        return q

    def count(self):
        """
        Perform a SELECT COUNT() and return the number of records as an
        integer.

        If the QuerySet is already fully cached, return the length of the
        cached results set to avoid multiple SELECT COUNT(*) calls.
        """
        if self._result_cache is not None:
            return len(self._result_cache)

        return self.query.get_count(using=self.db)

    def get(self, *args, **kwargs):
        """
        Perform the query and return a single object matching the given
        keyword arguments.
        """
        clone = self.filter(*args, **kwargs)
        if self.query.can_filter() and not self.query.distinct_fields:
            clone = clone.order_by()
        # print ('FROM GET', clone.query)
        # print (clone.query.__dict__)
        # clone = self.model.apply_policy(clone)
        num = len(clone)
        if num == 1:
            return clone._result_cache[0]
        if not num:
            raise self.model.DoesNotExist(
                "%s matching query does not exist." %
                self.model._meta.object_name
            )
        raise self.model.MultipleObjectsReturned(
            "get() returned more than one %s -- it returned %s!" %
            (self.model._meta.object_name, num)
        )

    def annotate(self, *args, **kwargs):
        """
        Return a query set in which the returned objects have been annotated
        with extra data or aggregations.
        """
        self._validate_values_are_expressions(args + tuple(kwargs.values()), method_name='annotate')
        annotations = OrderedDict()  # To preserve ordering of args
        for arg in args:
            # The default_alias property may raise a TypeError.
            try:
                if arg.default_alias in kwargs:
                    raise ValueError("The named annotation '%s' conflicts with the "
                                     "default name for another annotation."
                                     % arg.default_alias)
            except TypeError:
                raise TypeError("Complex annotations require an alias")
            annotations[arg.default_alias] = arg
        annotations.update(kwargs)

        clone = self._chain()
        names = self._fields
        if names is None:
            names = set(chain.from_iterable(
                (field.name, field.attname) if hasattr(field, 'attname') else (field.name,)
                for field in self.model._meta.get_fields()
            ))

        for alias, annotation in annotations.items():

            if alias in names:
                raise ValueError("The annotation '%s' conflicts with a field on "
                                 "the model." % alias)
            if isinstance(annotation, FilteredRelation):
                clone.query.add_filtered_relation(annotation, alias)
            else:
                clone.query.add_annotation(annotation, alias, is_summary=False)

        for alias, annotation in clone.query.annotations.items():
            if alias in annotations and annotation.contains_aggregate:
                if clone._fields is None:
                    clone.query.group_by = True
                else:
                    clone.query.set_group_by()
                break

        # clone = self.model.apply_policy(clone)
        # print ('ANOTTATE', clone.query)

        return clone

    def values(self, *fields, **expressions):
        fields += tuple(expressions)
        clone = self._values(*fields, **expressions)
        clone._iterable_class = ValuesIterable
        return clone

    def values_list(self, *fields, flat=False, named=False):
        if flat and named:
            raise TypeError("'flat' and 'named' can't be used together.")
        if flat and len(fields) > 1:
            raise TypeError("'flat' is not valid when values_list is called with more than one field.")

        field_names = {f for f in fields if not hasattr(f, 'resolve_expression')}
        _fields = []
        expressions = {}
        counter = 1
        for field in fields:
            if hasattr(field, 'resolve_expression'):
                field_id_prefix = getattr(field, 'default_alias', field.__class__.__name__.lower())
                while True:
                    field_id = field_id_prefix + str(counter)
                    counter += 1
                    if field_id not in field_names:
                        break
                expressions[field_id] = field
                _fields.append(field_id)
            else:
                _fields.append(field)

        clone = self._values(*_fields, **expressions)
        clone._iterable_class = (
            NamedValuesListIterable if named
            else FlatValuesListIterable if flat
            else ValuesListIterable
        )
        return clone

    def _prefetch_related_objects(self):
        # This method can only be called once the result cache has been filled.
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def prefetch_related(self, *lookups):
        """
        Return a new QuerySet instance that will prefetch the specified
        Many-To-One and Many-To-Many related objects when the QuerySet is
        evaluated.

        When prefetch_related() is called more than once, append to the list of
        prefetch lookups. If prefetch_related(None) is called, clear the list.
        """
        clone = self._chain()
        if lookups == (None,):
            clone._prefetch_related_lookups = ()
        else:
            for lookup in lookups:
                if isinstance(lookup, Prefetch):
                    lookup = lookup.prefetch_to
                lookup = lookup.split(LOOKUP_SEP, 1)[0]
                if lookup in self.query._filtered_relations:
                    raise ValueError('prefetch_related() is not supported with FilteredRelation.')
            clone._prefetch_related_lookups = clone._prefetch_related_lookups + lookups
        return clone

    def _fetch_all(self):
        if self._result_cache is None:
            self._result_cache = list(self._iterable_class(self))
        if self._prefetch_related_lookups and not self._prefetch_done:
            self._prefetch_related_objects()

class EstrelaManager(BaseManager.from_queryset(EstrelaQuerySet)):
    """The Estrela version of Django's Manager class, which [...]
    """
    pass
    # def get_queryset(self):
    #     policy_applied.append(str(self.model))
    #     return EstrelaQuerySet(self.model, using=self._db)

class EstrelaModel(models.Model):
    """ Estrela version of Django's Model class.
    """

    objects = EstrelaManager()
    _objects = Manager()

    def __init__(self, *args, **kw):
        super(EstrelaModel, self).__init__(*args, **kw)

    def __setattr__(self, name, value):
        field_names = [field.name for field in self._meta.concrete_fields] \
                        if hasattr(self, '_meta') else []
        models.Model.__setattr__(self, name, value)

    @classmethod
    @abstractmethod
    def apply_policy(cls, queryset, **agg_expr):
        return queryset

    @classmethod
    @abstractmethod
    def apply_post_policy(cls, rset, fields):
        return rset
    
    class Meta(object):
        """Abstract class.
        """
        abstract = True

class Aggregate(Func):
    contains_aggregate = True
    name = None

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # Aggregates are not allowed in UPDATE queries, so ignore for_save
        c = super(Aggregate, self).resolve_expression(query, allow_joins, reuse, summarize)
        if not summarize:
            expressions = c.get_source_expressions()
            for index, expr in enumerate(expressions):
                if expr.contains_aggregate:
                    before_resolved = self.get_source_expressions()[index]
                    name = before_resolved.name if hasattr(before_resolved, 'name') else repr(before_resolved)
                    raise FieldError("Cannot compute %s('%s'): '%s' is an aggregate" % (c.name, name, name))
        return c

    @property
    def default_alias(self):
        expressions = self.get_source_expressions()
        if len(expressions) == 1 and hasattr(expressions[0], 'name'):
            return '%s__%s' % (expressions[0].name, self.name.lower())
        raise TypeError("Complex expressions require an alias")

    def get_group_by_cols(self):
        return []


class Avg(Aggregate):
    function = 'AVG'
    name = 'Avg'

    def _resolve_output_field(self):
        source_field = self.get_source_fields()[0]
        if isinstance(source_field, (IntegerField, DecimalField)):
            self._output_field = FloatField()
        super(Avg, self)._resolve_output_field()

    def as_oracle(self, compiler, connection):
        if self.output_field.get_internal_type() == 'DurationField':
            expression = self.get_source_expressions()[0]
            from django.db.backends.oracle.functions import IntervalToSeconds, SecondsToInterval
            return compiler.compile(
                SecondsToInterval(Avg(IntervalToSeconds(expression)))
            )
        return super(Avg, self).as_sql(compiler, connection)


class Count(Aggregate):
    function = 'COUNT'
    name = 'Count'
    template = '%(function)s(%(distinct)s%(expressions)s)'

    def __init__(self, expression, distinct=False, **extra):
        if expression == '*':
            expression = Star()
        super(Count, self).__init__(
            expression, distinct='DISTINCT ' if distinct else '', output_field=IntegerField(), **extra)

    def __repr__(self):
        return "{}({}, distinct={})".format(
            self.__class__.__name__,
            self.arg_joiner.join(str(arg) for arg in self.source_expressions),
            'False' if self.extra['distinct'] == '' else 'True',
        )

    def convert_value(self, value, expression, connection, context):
        if value is None:
            return 0
        return int(value)


class Max(Aggregate):
    function = 'MAX'
    name = 'Max'


class Min(Aggregate):
    function = 'MIN'
    name = 'Min'


class StdDev(Aggregate):
    name = 'StdDev'

    def __init__(self, expression, sample=False, **extra):
        self.function = 'STDDEV_SAMP' if sample else 'STDDEV_POP'
        super(StdDev, self).__init__(expression, output_field=FloatField(), **extra)

    def __repr__(self):
        return "{}({}, sample={})".format(
            self.__class__.__name__,
            self.arg_joiner.join(str(arg) for arg in self.source_expressions),
            'False' if self.function == 'STDDEV_POP' else 'True',
        )

    def convert_value(self, value, expression, connection, context):
        if value is None:
            return value
        return float(value)


class Sum(Aggregate):
    function = 'SUM'
    name = 'Sum'

    def as_oracle(self, compiler, connection):
        if self.output_field.get_internal_type() == 'DurationField':
            expression = self.get_source_expressions()[0]
            from django.db.backends.oracle.functions import IntervalToSeconds, SecondsToInterval
            return compiler.compile(
                SecondsToInterval(Sum(IntervalToSeconds(expression)))
            )
        return super(Sum, self).as_sql(compiler, connection)


class Variance(Aggregate):
    name = 'Variance'

    def __init__(self, expression, sample=False, **extra):
        self.function = 'VAR_SAMP' if sample else 'VAR_POP'
        super(Variance, self).__init__(expression, output_field=FloatField(), **extra)

    def __repr__(self):
        return "{}({}, sample={})".format(
            self.__class__.__name__,
            self.arg_joiner.join(str(arg) for arg in self.source_expressions),
            'False' if self.function == 'VAR_POP' else 'True',
        )

    def convert_value(self, value, expression, connection, context):
        if value is None:
            return value
        return float(value)

def get_related_populators(klass_info, select, db):
    iterators = []
    related_klass_infos = klass_info.get('related_klass_infos', [])
    for rel_klass_info in related_klass_infos:
        rel_cls = RelatedPopulator(rel_klass_info, select, db)
        iterators.append(rel_cls)
    return iterators

class Prefetch:
    def __init__(self, lookup, queryset=None, to_attr=None):
        # `prefetch_through` is the path we traverse to perform the prefetch.
        self.prefetch_through = lookup
        # `prefetch_to` is the path to the attribute that stores the result.
        self.prefetch_to = lookup
        if queryset is not None and not issubclass(queryset._iterable_class, ModelIterable):
            raise ValueError('Prefetch querysets cannot use values().')
        if to_attr:
            self.prefetch_to = LOOKUP_SEP.join(lookup.split(LOOKUP_SEP)[:-1] + [to_attr])

        self.queryset = queryset
        self.to_attr = to_attr

    def __getstate__(self):
        obj_dict = self.__dict__.copy()
        if self.queryset is not None:
            # Prevent the QuerySet from being evaluated
            obj_dict['queryset'] = self.queryset._chain(
                _result_cache=[],
                _prefetch_done=True,
            )
        return obj_dict

    def add_prefix(self, prefix):
        self.prefetch_through = prefix + LOOKUP_SEP + self.prefetch_through
        self.prefetch_to = prefix + LOOKUP_SEP + self.prefetch_to

    def get_current_prefetch_to(self, level):
        return LOOKUP_SEP.join(self.prefetch_to.split(LOOKUP_SEP)[:level + 1])

    def get_current_to_attr(self, level):
        parts = self.prefetch_to.split(LOOKUP_SEP)
        to_attr = parts[level]
        as_attr = self.to_attr and level == len(parts) - 1
        return to_attr, as_attr

    def get_current_queryset(self, level):
        if self.get_current_prefetch_to(level) == self.prefetch_to:
            return self.queryset
        return None

    def __eq__(self, other):
        return isinstance(other, Prefetch) and self.prefetch_to == other.prefetch_to

    def __hash__(self):
        return hash((self.__class__, self.prefetch_to))

def normalize_prefetch_lookups(lookups, prefix=None):
    """Normalize lookups into Prefetch objects."""
    ret = []
    for lookup in lookups:
        if not isinstance(lookup, Prefetch):
            lookup = Prefetch(lookup)
        if prefix:
            lookup.add_prefix(prefix)
        ret.append(lookup)
    return ret


def prefetch_related_objects(model_instances, *related_lookups):
    """
    Populate prefetched object caches for a list of model instances based on
    the lookups/Prefetch instances given.
    """
    if not model_instances:
        return  # nothing to do

    # We need to be able to dynamically add to the list of prefetch_related
    # lookups that we look up (see below).  So we need some book keeping to
    # ensure we don't do duplicate work.
    done_queries = {}    # dictionary of things like 'foo__bar': [results]

    auto_lookups = set()  # we add to this as we go through.
    followed_descriptors = set()  # recursion protection

    all_lookups = normalize_prefetch_lookups(reversed(related_lookups))
    while all_lookups:
        lookup = all_lookups.pop()
        if lookup.prefetch_to in done_queries:
            if lookup.queryset:
                raise ValueError("'%s' lookup was already seen with a different queryset. "
                                 "You may need to adjust the ordering of your lookups." % lookup.prefetch_to)

            continue

        # Top level, the list of objects to decorate is the result cache
        # from the primary QuerySet. It won't be for deeper levels.
        obj_list = model_instances

        through_attrs = lookup.prefetch_through.split(LOOKUP_SEP)
        for level, through_attr in enumerate(through_attrs):
            # Prepare main instances
            if not obj_list:
                break

            prefetch_to = lookup.get_current_prefetch_to(level)
            if prefetch_to in done_queries:
                # Skip any prefetching, and any object preparation
                obj_list = done_queries[prefetch_to]
                continue

            # Prepare objects:
            good_objects = True
            for obj in obj_list:
                # Since prefetching can re-use instances, it is possible to have
                # the same instance multiple times in obj_list, so obj might
                # already be prepared.
                if not hasattr(obj, '_prefetched_objects_cache'):
                    try:
                        obj._prefetched_objects_cache = {}
                    except (AttributeError, TypeError):
                        # Must be an immutable object from
                        # values_list(flat=True), for example (TypeError) or
                        # a QuerySet subclass that isn't returning Model
                        # instances (AttributeError), either in Django or a 3rd
                        # party. prefetch_related() doesn't make sense, so quit.
                        good_objects = False
                        break
            if not good_objects:
                break

            # Descend down tree

            # We assume that objects retrieved are homogeneous (which is the premise
            # of prefetch_related), so what applies to first object applies to all.
            first_obj = obj_list[0]
            to_attr = lookup.get_current_to_attr(level)[0]
            prefetcher, descriptor, attr_found, is_fetched = get_prefetcher(first_obj, through_attr, to_attr)

            if not attr_found:
                raise AttributeError("Cannot find '%s' on %s object, '%s' is an invalid "
                                     "parameter to prefetch_related()" %
                                     (through_attr, first_obj.__class__.__name__, lookup.prefetch_through))

            if level == len(through_attrs) - 1 and prefetcher is None:
                # Last one, this *must* resolve to something that supports
                # prefetching, otherwise there is no point adding it and the
                # developer asking for it has made a mistake.
                raise ValueError("'%s' does not resolve to an item that supports "
                                 "prefetching - this is an invalid parameter to "
                                 "prefetch_related()." % lookup.prefetch_through)

            if prefetcher is not None and not is_fetched:
                obj_list, additional_lookups = prefetch_one_level(obj_list, prefetcher, lookup, level)
                # We need to ensure we don't keep adding lookups from the
                # same relationships to stop infinite recursion. So, if we
                # are already on an automatically added lookup, don't add
                # the new lookups from relationships we've seen already.
                if not (prefetch_to in done_queries and lookup in auto_lookups and descriptor in followed_descriptors):
                    done_queries[prefetch_to] = obj_list
                    new_lookups = normalize_prefetch_lookups(reversed(additional_lookups), prefetch_to)
                    auto_lookups.update(new_lookups)
                    all_lookups.extend(new_lookups)
                followed_descriptors.add(descriptor)
            else:
                # Either a singly related object that has already been fetched
                # (e.g. via select_related), or hopefully some other property
                # that doesn't support prefetching but needs to be traversed.

                # We replace the current list of parent objects with the list
                # of related objects, filtering out empty or missing values so
                # that we can continue with nullable or reverse relations.
                new_obj_list = []
                for obj in obj_list:
                    if through_attr in getattr(obj, '_prefetched_objects_cache', ()):
                        # If related objects have been prefetched, use the
                        # cache rather than the object's through_attr.
                        new_obj = list(obj._prefetched_objects_cache.get(through_attr))
                    else:
                        try:
                            new_obj = getattr(obj, through_attr)
                        except exceptions.ObjectDoesNotExist:
                            continue
                    if new_obj is None:
                        continue
                    # We special-case `list` rather than something more generic
                    # like `Iterable` because we don't want to accidentally match
                    # user models that define __iter__.
                    if isinstance(new_obj, list):
                        new_obj_list.extend(new_obj)
                    else:
                        new_obj_list.append(new_obj)
                obj_list = new_obj_list

def get_prefetcher(instance, through_attr, to_attr):
    """
    For the attribute 'through_attr' on the given instance, find
    an object that has a get_prefetch_queryset().
    Return a 4 tuple containing:
    (the object with get_prefetch_queryset (or None),
     the descriptor object representing this relationship (or None),
     a boolean that is False if the attribute was not found at all,
     a boolean that is True if the attribute has already been fetched)
    """
    prefetcher = None
    is_fetched = False

    # For singly related objects, we have to avoid getting the attribute
    # from the object, as this will trigger the query. So we first try
    # on the class, in order to get the descriptor object.
    rel_obj_descriptor = getattr(instance.__class__, through_attr, None)
    if rel_obj_descriptor is None:
        attr_found = hasattr(instance, through_attr)
    else:
        attr_found = True
        if rel_obj_descriptor:
            # singly related object, descriptor object has the
            # get_prefetch_queryset() method.
            if hasattr(rel_obj_descriptor, 'get_prefetch_queryset'):
                prefetcher = rel_obj_descriptor
                if rel_obj_descriptor.is_cached(instance):
                    is_fetched = True
            else:
                # descriptor doesn't support prefetching, so we go ahead and get
                # the attribute on the instance rather than the class to
                # support many related managers
                rel_obj = getattr(instance, through_attr)
                if hasattr(rel_obj, 'get_prefetch_queryset'):
                    prefetcher = rel_obj
                if through_attr != to_attr:
                    # Special case cached_property instances because hasattr
                    # triggers attribute computation and assignment.
                    if isinstance(getattr(instance.__class__, to_attr, None), cached_property):
                        is_fetched = to_attr in instance.__dict__
                    else:
                        is_fetched = hasattr(instance, to_attr)
                else:
                    is_fetched = through_attr in instance._prefetched_objects_cache
    return prefetcher, rel_obj_descriptor, attr_found, is_fetched


def prefetch_one_level(instances, prefetcher, lookup, level):
    """
    Helper function for prefetch_related_objects().

    Run prefetches on all instances using the prefetcher object,
    assigning results to relevant caches in instance.

    Return the prefetched objects along with any additional prefetches that
    must be done due to prefetch_related lookups found from default managers.
    """
    # prefetcher must have a method get_prefetch_queryset() which takes a list
    # of instances, and returns a tuple:

    # (queryset of instances of self.model that are related to passed in instances,
    #  callable that gets value to be matched for returned instances,
    #  callable that gets value to be matched for passed in instances,
    #  boolean that is True for singly related objects,
    #  cache or field name to assign to,
    #  boolean that is True when the previous argument is a cache name vs a field name).

    # The 'values to be matched' must be hashable as they will be used
    # in a dictionary.

    rel_qs, rel_obj_attr, instance_attr, single, cache_name, is_descriptor = (
        prefetcher.get_prefetch_queryset(instances, lookup.get_current_queryset(level)))
    # We have to handle the possibility that the QuerySet we just got back
    # contains some prefetch_related lookups. We don't want to trigger the
    # prefetch_related functionality by evaluating the query. Rather, we need
    # to merge in the prefetch_related lookups.
    # Copy the lookups in case it is a Prefetch object which could be reused
    # later (happens in nested prefetch_related).
    additional_lookups = [
        copy.copy(additional_lookup) for additional_lookup
        in getattr(rel_qs, '_prefetch_related_lookups', ())
    ]
    if additional_lookups:
        # Don't need to clone because the manager should have given us a fresh
        # instance, so we access an internal instead of using public interface
        # for performance reasons.
        rel_qs._prefetch_related_lookups = ()

    all_related_objects = list(rel_qs)

    rel_obj_cache = {}
    for rel_obj in all_related_objects:
        rel_attr_val = rel_obj_attr(rel_obj)
        rel_obj_cache.setdefault(rel_attr_val, []).append(rel_obj)

    to_attr, as_attr = lookup.get_current_to_attr(level)
    # Make sure `to_attr` does not conflict with a field.
    if as_attr and instances:
        # We assume that objects retrieved are homogeneous (which is the premise
        # of prefetch_related), so what applies to first object applies to all.
        model = instances[0].__class__
        try:
            model._meta.get_field(to_attr)
        except exceptions.FieldDoesNotExist:
            pass
        else:
            msg = 'to_attr={} conflicts with a field on the {} model.'
            raise ValueError(msg.format(to_attr, model.__name__))

    # Whether or not we're prefetching the last part of the lookup.
    leaf = len(lookup.prefetch_through.split(LOOKUP_SEP)) - 1 == level

    for obj in instances:
        instance_attr_val = instance_attr(obj)
        vals = rel_obj_cache.get(instance_attr_val, [])

        if single:
            val = vals[0] if vals else None
            if as_attr:
                # A to_attr has been given for the prefetch.
                setattr(obj, to_attr, val)
            elif is_descriptor:
                # cache_name points to a field name in obj.
                # This field is a descriptor for a related object.
                setattr(obj, cache_name, val)
            else:
                # No to_attr has been given for this prefetch operation and the
                # cache_name does not point to a descriptor. Store the value of
                # the field in the object's field cache.
                obj._state.fields_cache[cache_name] = val
        else:
            if as_attr:
                setattr(obj, to_attr, vals)
            else:
                manager = getattr(obj, to_attr)
                if leaf and lookup.queryset is not None:
                    qs = manager._apply_rel_filters(lookup.queryset)
                else:
                    qs = manager.get_queryset()
                qs._result_cache = vals
                # We don't want the individual qs doing prefetch_related now,
                # since we have merged this into the current work.
                qs._prefetch_done = True
                obj._prefetched_objects_cache[cache_name] = qs
    return all_related_objects, additional_lookups
