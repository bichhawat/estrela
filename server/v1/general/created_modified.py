from EstrelaModel import EstrelaModel
from django.db import models


class CreatedModified(EstrelaModel):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
