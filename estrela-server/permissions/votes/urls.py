from django.conf.urls import url
from permissions.permissions import full_permissions


urlpatterns = [

    # Post votes
    url(r'^post_votes$', full_permissions),
    url(r'^post_votes/(?P<post_vote_id>[\d]+)$', full_permissions),

]
