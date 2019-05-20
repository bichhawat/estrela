from django.conf.urls import url
from permissions.permissions import full_permissions


urlpatterns = [

    # Post replies
    url(r'^post_replies$', full_permissions),
    url(r'^post_replies/(?P<post_reply_id>[\d]+)$', full_permissions),

]
