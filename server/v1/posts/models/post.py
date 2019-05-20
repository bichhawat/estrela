from django.conf import settings
from django.db import models
from v1.general.created_modified import CreatedModified
from django.db.models import Q

exposed_request = None

class Post(CreatedModified):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    body = models.TextField()
    image = models.ImageField(blank=True)

    class Meta:
        default_related_name = 'posts'

    @classmethod
    def apply_policy(cls, queryset):
        return queryset

    @classmethod
    def apply_post_policy(cls, resultset, fields):
        ruser = exposed_request.user
        if not ruser.is_anonymous:
            following_users = ruser.following.values_list('id', flat=True).distinct()
            for row in resultset:
                user_id = row.user_id
                if user_id in following_users or user_id == ruser:
                    pass
                else:
                    row.body = '[REDACTED] Follow user to see the contents of this post'
        else:
            resultset = []
            
        return resultset

    def __str__(self):
        return self.title
