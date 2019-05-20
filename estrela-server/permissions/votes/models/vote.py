from django.conf import settings
from django.db import models

# Votes
VOTE_UP = 1
VOTE_DOWN = -1
VOTE_VALUE_CHOICES = (
    (VOTE_UP, 'Up'),
    (VOTE_DOWN, 'Down')
)

class Vote(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    value = models.IntegerField(choices=VOTE_VALUE_CHOICES)

    class Meta:
        abstract = True
