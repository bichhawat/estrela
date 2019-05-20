from django.conf import settings
from django.db import models
from EstrelaModel import EstrelaModel

class Profile(EstrelaModel):
    image = models.ImageField(blank=True)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    def __str__(self):
        return self.user.email
