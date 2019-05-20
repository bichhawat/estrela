from django.conf import settings
from django.db import models
from v1.utils import constants
from EstrelaModel import EstrelaModel

class Vote(EstrelaModel):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,)
    value = models.IntegerField(choices=constants.VOTE_VALUE_CHOICES)

    class Meta:
        abstract = True
