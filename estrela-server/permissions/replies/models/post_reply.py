from django.db import models
from permissions.posts.models.post import Post
from .reply import Reply


class PostReply(Reply):
    post = models.ForeignKey(Post, on_delete=models.CASCADE)

    class Meta:
        default_related_name = 'post_replies'

    def __str__(self):
        return self.body
